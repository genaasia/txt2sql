import os
import mimetypes
import re

import httpx
import tqdm

from urllib.parse import unquote, urlparse


def extract_file_id_from_google_drive_link(link):
    """
    Extracts the file ID from a Google Drive link.

    Args:
        link (str): Google Drive link

    Returns:
        str: File ID
    """
    # Handle different forms of Google Drive links
    if "drive.google.com/file/d/" in link:
        # Format: https://drive.google.com/file/d/FILE_ID/view
        file_id = link.split("/file/d/")[1].split("/")[0]
    elif "drive.google.com/open?id=" in link:
        # Format: https://drive.google.com/open?id=FILE_ID
        file_id = link.split("open?id=")[1]
    elif "drive.google.com/uc?export=download&id=" in link:
        # Format: https://drive.google.com/uc?export=download&id=FILE_ID
        file_id = link.split("id=")[1].split("&")[0]
    else:
        raise ValueError("Unsupported Google Drive link format")

    return file_id


def download_from_google_drive(download_url: str, base_path: str):
    """
    Downloads a file from Google Drive using its file ID with a progress bar.
    Handles large files that trigger Google Drive's virus scan warning.

    Args:
        download_url (str): The url from the Google Drive link
        base_path (str): Local directory where the file will be saved.

    Returns:
        str: Path to the downloaded file
    """
    # Get file id
    file_id = extract_file_id_from_google_drive_link(download_url)
    # Initial URL to get filename and confirmation token
    initial_url = f"https://drive.google.com/uc?id={file_id}&export=download"

    with httpx.Client(follow_redirects=True) as client:
        # First request to get cookies/confirmation token and filename
        response = client.get(initial_url)

        # Try to extract the original filename
        original_filename = None

        # Check content-disposition header first
        cd_header = response.headers.get("content-disposition")
        if cd_header:
            filename_match = re.search(r'filename="(.+?)"', cd_header)
            if filename_match:
                original_filename = unquote(filename_match.group(1))

        # If not found in headers, try to extract from HTML
        if not original_filename and "Virus scan warning" in response.text:
            # Look for the filename in the HTML content
            filename_match = re.search(r'<span class="uc-name-size"><a[^>]*>([^<]+)</a>', response.text)
            if filename_match:
                original_filename = filename_match.group(1)

        # If still no filename found, use the default
        if not original_filename:
            original_filename = f"{file_id}.zip"

        # If destination was not specified, use the original filename
        destination = os.path.join(base_path, original_filename)
        if os.path.exists(destination):
            print(f"File already exists at destination: {destination}")
            return destination
        print(f"Downloading: {destination}")

        # Check if we received a virus scan warning page
        confirm_token = None
        if "Virus scan warning" in response.text:
            # Extract the confirmation token
            match = re.search(r'name="confirm" value="([^"]+)"', response.text)
            if match:
                confirm_token = match.group(1)

        # Download URL with or without confirmation
        if confirm_token:
            download_url = (
                f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm={confirm_token}"
            )
        else:
            download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"

        # Stream the download with the confirmation token if necessary
        with client.stream("GET", download_url, follow_redirects=True) as response:
            # Get the total file size if available
            total_size = int(response.headers.get("content-length", 0))

            # Check if content is HTML (error) or the actual file
            content_type = response.headers.get("content-type", "")
            if "text/html" in content_type and total_size < 10000:
                # This is likely an error page, not the file
                html_content = b"".join(response.iter_bytes())
                with open("error_page.html", "wb") as f:
                    f.write(html_content)
                raise ValueError("Failed to download file. Error page saved to 'error_page.html'")

            # Initialize the progress bar
            progress_bar = tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(destination))

            # Write the file to the destination with progress bar
            with open(destination, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            # Close the progress bar
            progress_bar.close()

    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        print(f"File downloaded successfully to {destination}")
        return destination
    else:
        print("Download appears to have failed. Check the file.")
        return None


def download_from_url(download_url: str, base_path: str):
    """
    Downloads a file from any generic URL with a progress bar.

    Args:
        url (str): URL to download
        base_path (str): Local directory where the file will be saved.

    Returns:
        str: Path to the downloaded file
    """
    with httpx.Client(follow_redirects=True) as client:
        # Send a HEAD request first to get headers without downloading content
        head_response = client.head(download_url)

        # Try to get the filename from content-disposition header
        original_filename = None
        cd_header = head_response.headers.get("content-disposition")
        if cd_header:
            filename_match = re.search(r'filename="?([^"]+)"?', cd_header)
            if filename_match:
                original_filename = unquote(filename_match.group(1))

        # If no filename in header, extract from URL
        if not original_filename:
            parsed_url = urlparse(download_url)
            path = parsed_url.path
            original_filename = os.path.basename(path)

            # If the URL path ends with a slash or is empty, use a default name
            if not original_filename:
                # Try to determine extension from content-type
                content_type = head_response.headers.get("content-type", "")
                ext = mimetypes.guess_extension(content_type) or ".bin"
                original_filename = f"download{ext}"

        # Remove query parameters from filename if they exist
        if "?" in original_filename:
            original_filename = original_filename.split("?")[0]

        # Use original filename if destination is not specified
        destination = os.path.join(base_path, original_filename)
        if os.path.exists(destination):
            print(f"File already exists at destination: {destination}")
            return destination
        print(f"Downloading: {original_filename}")

        # Stream the actual file
        with client.stream("GET", download_url) as response:
            # Get the total file size if available
            total_size = int(response.headers.get("content-length", 0))

            # Initialize the progress bar
            progress_bar = tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(destination))

            # Write the file to the destination with progress bar
            with open(destination, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))

            # Close the progress bar
            progress_bar.close()

    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        print(f"File downloaded successfully to {destination}")
        return destination
    else:
        print("Download appears to have failed. Check the file.")
        return None


def download_link(download_url: str, base_path: str):
    if "google.com" in download_url:
        return download_from_google_drive(download_url, base_path)
    else:
        return download_from_url(download_url, base_path)
