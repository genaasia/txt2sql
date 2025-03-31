import os
import json
import shutil
import zipfile

from abc import ABC, abstractmethod

from txt2sql.data.datasets import SqliteDataset
from txt2sql.data.utils.download import download_link


BIRD_TRAIN_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
BULL_DOWNLOAD_URL = "https://drive.google.com/file/d/1OtyFdH9cs-6bEVj8yKK4Zt53N52L_dBH/view"
SPIDER_DOWNLOAD_URL = "https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view"


class DataLoader(ABC):
    def __init__(self, base_path: str):
        if not os.path.isdir(base_path):
            raise NotADirectoryError(f"base_path '{base_path}' is not a directory")
        self.base_path = base_path
        self.splits: dict[str, list] = dict()

    @classmethod
    @abstractmethod
    def download(cls):
        pass

    def list_splits(self) -> list[str]:
        return sorted(list(self.splits.keys()))

    def _postprocess(self, sample: dict, include_metadata: bool = False) -> dict:
        sample = sample.copy()
        if not include_metadata:
            sample.pop("metadata", None)
        return sample

    def get_split(self, split: str, include_metadata: bool = False) -> list[dict]:
        if split not in self.list_splits():
            raise ValueError(f"split '{split}' not supported")
        return [self._postprocess(sample, include_metadata=include_metadata) for sample in self.splits[split]]

    def get_sample(self, split: str, question_id: int, include_metadata: bool = False) -> dict:
        samples = self.splits[split]
        for sample in samples:
            if sample["question_id"] == question_id:
                return self._postprocess(sample, include_metadata=include_metadata)
        raise ValueError(f"question_id '{question_id}' not found in split '{split}'")

    @abstractmethod
    def get_dataset(self, split: str) -> SqliteDataset:
        pass


class BirdDataLoader(DataLoader):

    dset_info = {
        "train": {
            "url": BIRD_TRAIN_URL,
            "subdir": "train",
            "expected_dbs": 69,
        },
        "dev": {
            "url": BIRD_DEV_URL,
            "subdir": "dev_20240627",
            "expected_dbs": 11,
        },
    }

    def __init__(self, base_path: str):
        super().__init__(base_path)
        # check base_path exists, if not, prompt to download
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            print(
                "data not loaded, please download first using the download classmethod BirdDataLoader.download(<base_path>)"
            )
        self.check_data()
        self.splits: dict[str, list] = self._load_data()

    @classmethod
    def download(cls, base_path: str, train_url: str | None = None, dev_url: str | None = None):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        cls.download_split(base_path, "train", train_url)
        cls.download_split(base_path, "dev", dev_url)
        print(f"done, all datasets verified in '{base_path}'")

    @classmethod
    def download_split(cls, base_path: str, split: str, url: str | None):
        if not url:
            url = cls.dset_info[split]["url"]
        split_subdir = cls.dset_info[split]["subdir"]
        # first check all data
        split_ok, _ = cls.check_subset(base_path, split)
        if split_ok:
            print(f"verified all {split} data already exists, skipping")
            return
        print(f"downloading {split} dataset")
        zipfile_path = download_link(url, base_path)
        data_dir = os.path.join(base_path, split_subdir)
        split_db_dir = f"{split}_databases"
        dset_dir = os.path.join(data_dir, split_db_dir)
        if os.path.exists(data_dir) and os.path.exists(dset_dir):
            print(f"{split} directory and {split} database directories already exist, skipping")
            return
        print(f"unzipping {split} dataset")
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extractall(base_path)
        print(f"unzipping {split}_databases")
        db_zipfile = os.path.join(base_path, split_subdir, f"{split}_databases.zip")
        with zipfile.ZipFile(db_zipfile, "r") as zip_ref:
            zip_ref.extractall(os.path.join(base_path, split_subdir))
        # final confirmation
        split_ok, _ = cls.check_subset(base_path, split)
        if not split_ok:
            raise ValueError(f"failed to download {split} dataset")
        print(f"train {split} downloaded, unzipped and verified")
        return

    @classmethod
    def check_subset(self, base_path: str, split: str) -> tuple[bool, str]:
        subset_dir: str = self.dset_info[split]["subdir"]
        expected = self.dset_info[split]["expected_dbs"]
        if not os.path.exists(base_path):
            return False, f"base_path '{base_path}' does not exist"
        # check subset_dir exists
        data_path = os.path.join(base_path, subset_dir)
        if not os.path.exists(data_path):
            return False, f"subdirectory '{subset_dir}' does not exist"
        json_name = subset_dir.split("_")[0] + ".json"
        if not os.path.exists(os.path.join(data_path, json_name)):
            return False, f"data '{json_name}' does not exist in '{subset_dir}' directory"
        # check databases exist
        databases_subdir = subset_dir.split("_")[0] + "_databases"
        dset_path = os.path.join(data_path, databases_subdir)
        if not os.path.exists(dset_path):
            return False, f"database directory '{dset_path}' does not exist in '{subset_dir}' directory"
        # check that expected eleven subdirectories exist in dev_databases
        database_subdirs = [d for d in os.listdir(dset_path) if os.path.isdir(os.path.join(dset_path, d))]
        if len(database_subdirs) != expected:
            return False, f"expected {expected} subdirectories in '{dset_path}', found {len(database_subdirs)}"
        # for each dset dir, check that expected *.sqlite file exists
        for db_dir in database_subdirs:
            expected_sqlite = db_dir + ".sqlite"
            if not os.path.exists(os.path.join(dset_path, db_dir, expected_sqlite)):
                return False, f"expected '{expected_sqlite}' in '{db_dir}' directory"
        return True, "all checks passed"

    def check_data(self):
        all_ok = True
        for split in self.dset_info:
            split_ok, split_msg = self.check_subset(self.base_path, split)
            all_ok = split_ok and all_ok
            if not split_ok:
                print(f"{split} failed verification: {split_msg}")
        if not all_ok:
            raise ValueError("one or more datasets failed verification")
        return all_ok

    def _process_split(self, samples: list[dict]) -> list[dict]:
        formatted_samples: list[dict] = list()
        for idx, sample in enumerate(samples):
            formatted_sample = dict()
            if "question_id" in sample:
                formatted_sample["question_id"] = sample["question_id"]
            else:
                formatted_sample["question_id"] = idx
            formatted_sample["db_id"] = sample["db_id"]
            formatted_sample["question"] = sample["question"]
            formatted_sample["evidence"] = sample["evidence"]
            formatted_sample["sql"] = sample["SQL"]
            metadata = dict()
            for key, value in sample.items():
                if key not in formatted_sample:
                    metadata[key] = value
            formatted_sample["metadata"] = metadata
            formatted_samples.append(formatted_sample)
        return formatted_samples

    def _load_data(self):
        # load train.json from train directory
        train_json_path = os.path.join(self.base_path, self.dset_info["train"]["subdir"], "train.json")
        with open(train_json_path, "r") as f:
            train_raw_data = json.load(f)
        train_data = self._process_split(train_raw_data)
        # load dev.json from dev directory
        dev_json_path = os.path.join(self.base_path, self.dset_info["dev"]["subdir"], "dev.json")
        with open(dev_json_path, "r") as f:
            dev_raw_data = json.load(f)
        dev_data = self._process_split(dev_raw_data)
        return {"train": train_data, "dev": dev_data}

    def get_dataset(self, split: str) -> SqliteDataset:
        if split not in self.list_splits():
            raise ValueError(f"split '{split}' not supported")
        split_dir: str = self.dset_info[split]["subdir"]
        db_dir = split_dir.split("_")[0] + "_databases"
        db_path = os.path.join(self.base_path, split_dir, db_dir)
        if not os.path.isdir(db_path):
            raise NotADirectoryError(f"database directory '{db_path}' does not exist")
        dataset = SqliteDataset(base_data_path=db_path)
        # check that the dataset has the expected number of databases
        if len(dataset.get_databases()) != self.dset_info[split]["expected_dbs"]:
            raise ValueError(
                f"expected {self.dset_info[split]['expected_dbs']} databases, found {len(dataset.get_databases())}"
            )
        return dataset


class SpiderDataLoader(DataLoader):

    dset_default_url = SPIDER_DOWNLOAD_URL
    database_count = 166
    test_database_count = 206

    def __init__(self, base_path: str):
        super().__init__(base_path)
        # check base_path exists, if not, prompt to download
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            print(
                "data not loaded, please download first using the download classmethod SpiderDataLoader.download(<base_path>)"
            )
        self.check_data()
        self.splits: dict[str, list] = self._load_data()

    @classmethod
    def download(cls, base_path: str, url: str | None = None):
        # first, check if all datasets already exist
        try:
            cls.check_spider(base_path)
            print("verified all datasets already exist, skipping download")
            return
        except ValueError:
            pass
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        # download the zip file
        if not url:
            url = cls.dset_default_url
        zipfile_path = download_link(url, base_path)
        # unzip the file
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extractall(base_path)
        # move all contents from extracted directory to base_dir
        extracted_dir = os.path.join(base_path, "spider_data")
        if not os.path.isdir(extracted_dir):
            raise ValueError(f"expected 'spider_data' directory in '{base_path}'")
        # move contents to base_path
        for item in os.listdir(extracted_dir):
            shutil.move(os.path.join(extracted_dir, item), os.path.join(base_path, item))
        # final confirmation
        cls.check_spider(base_path)
        print(f"done, data verified in '{base_path}'")
        return

    @classmethod
    def check_spider(self, base_path: str) -> tuple[bool, str]:
        # check dev.json, test.json, train_spider.json and train_others.json exist in base_path
        if not os.path.exists(base_path):
            raise ValueError(f"base_path '{base_path}' does not exist")
        for fname in ["dev.json", "test.json", "train_spider.json", "train_others.json"]:
            if not os.path.isfile(os.path.join(base_path, fname)):
                raise ValueError(f"data '{fname}' does not exist in '{base_path}' directory")
        # check 'database' directory exists and has expected count of subdirectories
        if not os.path.exists(os.path.join(base_path, "database")):
            raise ValueError(f"database directory does not exist in '{base_path}' directory")
        database_subdirs = [
            d
            for d in os.listdir(os.path.join(base_path, "database"))
            if os.path.isdir(os.path.join(base_path, "database", d))
        ]
        if len(database_subdirs) != self.database_count:
            raise ValueError(
                f"expected {self.database_count} subdirectories in 'database', found {len(database_subdirs)}"
            )
        # check 'test_database' directory exists and has expected count of subdirectories
        if not os.path.exists(os.path.join(base_path, "test_database")):
            raise ValueError(f"test_database directory does not exist in '{base_path}' directory")
        test_database_subdirs = [
            d
            for d in os.listdir(os.path.join(base_path, "test_database"))
            if os.path.isdir(os.path.join(base_path, "test_database", d))
        ]
        if len(test_database_subdirs) != self.test_database_count:
            raise ValueError(
                f"expected {self.test_database_count} subdirectories in 'test_database', found {len(test_database_subdirs)}"
            )
        # for database subdir in database, check that expected *.sqlite file exists
        for db_dir in database_subdirs:
            expected_sqlite = db_dir + ".sqlite"
            if not os.path.exists(os.path.join(base_path, "database", db_dir, expected_sqlite)):
                raise ValueError(f"expected '{expected_sqlite}' in '{db_dir}' directory")
        # for database subdir in test_database, check that expected *.sqlite file exists
        for db_dir in test_database_subdirs:
            expected_sqlite = db_dir + ".sqlite"
            if not os.path.exists(os.path.join(base_path, "test_database", db_dir, expected_sqlite)):
                raise ValueError(f"expected '{expected_sqlite}' in '{db_dir}' directory")
        return True

    def check_data(self):
        self.check_spider(self.base_path)
        return True

    def _process_split(self, samples: list[dict]) -> list[dict]:
        formatted_samples: list[dict] = list()
        for idx, sample in enumerate(samples):
            formatted_sample = dict()
            formatted_sample["question_id"] = idx
            formatted_sample["db_id"] = sample["db_id"]
            formatted_sample["question"] = sample["question"]
            formatted_sample["sql"] = sample["query"]
            metadata = dict()
            for key, value in sample.items():
                if key not in formatted_sample:
                    metadata[key] = value
            formatted_sample["metadata"] = metadata
            formatted_samples.append(formatted_sample)
        return formatted_samples

    def _load_data(self) -> list[dict]:
        # train_spider.json as train
        train_spider_json_path = os.path.join(self.base_path, "train_spider.json")
        with open(train_spider_json_path, "r") as f:
            train_spider_raw_data = json.load(f)
        train_spider_data = self._process_split(train_spider_raw_data)
        # train_others.json as train-other
        train_others_json_path = os.path.join(self.base_path, "train_others.json")
        with open(train_others_json_path, "r") as f:
            train_others_raw_data = json.load(f)
        train_others_data = self._process_split(train_others_raw_data)
        # dev.json as dev
        dev_json_path = os.path.join(self.base_path, "dev.json")
        with open(dev_json_path, "r") as f:
            dev_raw_data = json.load(f)
        dev_data = self._process_split(dev_raw_data)
        # test.json as test
        test_json_path = os.path.join(self.base_path, "test.json")
        with open(test_json_path, "r") as f:
            test_raw_data = json.load(f)
        test_data = self._process_split(test_raw_data)
        return {"train": train_spider_data, "train-other": train_others_data, "dev": dev_data, "test": test_data}

    def get_dataset(self, split: str) -> SqliteDataset:
        if split not in self.list_splits():
            raise ValueError(f"split '{split}' not supported")
        if split == "test":
            db_dir = "test_database"
            exp_count = self.test_database_count
        else:
            db_dir = "database"
            exp_count = self.database_count
        db_path = os.path.join(self.base_path, db_dir)
        if not os.path.isdir(db_path):
            raise NotADirectoryError(f"database directory '{db_path}' does not exist")
        dataset = SqliteDataset(base_data_path=db_path)
        # check that the dataset has the expected number of databases
        if len(dataset.get_databases()) != exp_count:
            raise ValueError(f"expected {exp_count} databases, found {len(dataset.get_databases())}")
        return dataset


class BullDataLoader(DataLoader):

    dset_default_url = BULL_DOWNLOAD_URL
    database_count = 3

    def __init__(self, base_path: str, language: str = "en", rename_bad_files: bool = True):
        super().__init__(base_path)
        if language not in ["en", "cn"]:
            raise ValueError(f"language '{language}' not supported, only 'en' and 'cn' supported")
        if language == "en":
            # print warning that databases are not correct for en
            print(
                "WARNING: the BULL English datasets are not translated! Execution on the databases will not work as expected."
            )
        self.language = language
        self.rename_bad_files = rename_bad_files
        # check base_path exists, if not, prompt to download
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            print(
                "data not loaded, please download first using the download classmethod BullDataLoader.download(<base_path>)"
            )
        self.check_data()
        self.splits: dict[str, list[dict]] = self._load_data()

    @classmethod
    def download(cls, base_path: str, url: str | None = None, rename_bad_files: bool = True):
        # first, check if all datasets already exist
        try:
            cls.check_bull(base_path, rename_bad_files=rename_bad_files)
            print("verified all datasets already exist, skipping download")
            return
        except ValueError:
            pass
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        # download the zip file
        if not url:
            url = cls.dset_default_url
        zipfile_path = download_link(url, base_path)
        # unzip the file
        with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
            zip_ref.extractall(base_path)
        # move all contents from extracted directory to base_dir
        extracted_dir = os.path.join(base_path, "BULL")
        if not os.path.isdir(extracted_dir):
            raise ValueError(f"expected 'BULL' directory in '{base_path}'")
        # move contents to base_path
        for item in os.listdir(extracted_dir):
            shutil.move(os.path.join(extracted_dir, item), os.path.join(base_path, item))
        # rename all sqlite files starting with "[" to *.sqlite.backup
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.startswith("[") and file.endswith(".sqlite"):
                    os.rename(os.path.join(root, file), os.path.join(root, file + ".backup"))
        # final confirmation
        cls.check_bull(base_path, rename_bad_files=rename_bad_files)
        print(f"done, data verified in '{base_path}'")
        return

    @classmethod
    def check_bull(self, base_path: str, rename_bad_files: bool) -> tuple[bool, str]:
        if not os.path.exists(base_path):
            raise ValueError(f"base_path '{base_path}' does not exist")
        # for language in en, cn, check dir 'BULL-<lang> exists
        for lang in ["en", "cn"]:
            lang_dir = f"BULL-{lang}"
            if not os.path.exists(os.path.join(base_path, lang_dir)):
                raise ValueError(f"expected '{lang_dir}' directory in '{base_path}'")
            # check files db_info.json, dev_<lang>.json, tables.json, train.json exist
            for fname in ["db_info.json", f"dev_{lang}.json", "tables.json", "train.json"]:
                if not os.path.isfile(os.path.join(base_path, lang_dir, fname)):
                    raise ValueError(f"data '{fname}' does not exist in '{lang_dir}' directory")
        # for language in en, cn, check database dir 'database_<lang> exists
        for lang in ["en", "cn"]:
            db_dir = f"database_{lang}"
            if not os.path.exists(os.path.join(base_path, db_dir)):
                raise ValueError(f"expected '{db_dir}' directory in '{base_path}'")
            # check expected count of subdirectories
            database_subdirs = [
                d
                for d in os.listdir(os.path.join(base_path, db_dir))
                if os.path.isdir(os.path.join(base_path, db_dir, d))
            ]
            if len(database_subdirs) != self.database_count:
                raise ValueError(
                    f"expected {self.database_count} subdirectories in '{db_dir}', found {len(database_subdirs)}"
                )
            # for each dset dir, check that expected *.sqlite file exists
            for db_subdir in database_subdirs:
                expected_sqlite = db_subdir + ".sqlite"
                if not os.path.exists(os.path.join(base_path, db_dir, db_subdir, expected_sqlite)):
                    raise ValueError(f"expected '{expected_sqlite}' in '{os.path.join(db_dir, db_subdir)}' directory")
        # print warning if any *.sqlite files starting with "["
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.startswith("[") and file.endswith(".sqlite"):
                    if rename_bad_files:
                        print(f"WARNING: found unexpected file '{file}' in '{root}', renaming to '{file}.backup'")
                        os.rename(os.path.join(root, file), os.path.join(root, file + ".backup"))
                    else:
                        print(
                            f"WARNING: found unexpected file '{file}' in '{root}', recommend renaming to '{file}.backup'"
                        )
        return True

    def check_data(self):
        self.check_bull(self.base_path, self.rename_bad_files)
        return True

    def _process_split(self, samples: list[dict]) -> list[dict]:
        formatted_samples: list[dict] = list()
        for idx, sample in enumerate(samples):
            formatted_sample = dict()
            formatted_sample["question_id"] = sample["q_id"]
            formatted_sample["db_id"] = sample["db_name"]
            formatted_sample["question"] = sample["question"]
            formatted_sample["sql"] = sample["sql_query"]
            metadata = dict()
            for key, value in sample.items():
                if key not in formatted_sample:
                    metadata[key] = value
            formatted_sample["metadata"] = metadata
            formatted_samples.append(formatted_sample)
        return formatted_samples

    def _load_data(self) -> dict[list[dict]]:
        data = dict()
        data["train"] = dict()
        data["dev"] = dict()
        # train.json as train
        train_json_path = os.path.join(self.base_path, f"BULL-{self.language}", "train.json")
        with open(train_json_path, "r") as f:
            train_raw_data = json.load(f)
        train_data = self._process_split(train_raw_data)
        # dev.json as dev
        dev_json_path = os.path.join(self.base_path, f"BULL-{self.language}", "dev.json")
        with open(dev_json_path, "r") as f:
            dev_raw_data = json.load(f)
        dev_data = self._process_split(dev_raw_data)
        return {"train": train_data, "dev": dev_data}

    def get_dataset(self, split: str) -> SqliteDataset:
        if split not in self.list_splits():
            raise ValueError(f"split '{split}' not supported")
        if self.language == "en":
            print(
                "WARNING: the BULL English datasets are not translated! Execution on the databases will not work as expected."
            )
        db_dir = f"database_{self.language}"
        db_path = os.path.join(self.base_path, db_dir)
        if not os.path.isdir(db_path):
            raise NotADirectoryError(f"database directory '{db_path}' does not exist")
        dataset = SqliteDataset(base_data_path=db_path)
        # check that the dataset has the expected number of databases
        if len(dataset.get_databases()) != self.database_count:
            raise ValueError(
                f"expected {self.database_count} databases, found {len(dataset.get_databases())}: {dataset.get_databases()}"
            )
        return dataset
