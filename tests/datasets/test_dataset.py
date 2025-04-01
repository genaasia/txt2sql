import os
import pytest
from txt2sql.data.datasets import SqliteDataset


class TestSqliteDataset:
    @pytest.fixture
    def sqlite_dataset(self):
        """Create a SqliteDataset instance pointing to the test data directory."""
        # Path to the directory containing the test database directories
        base_dir = os.path.join("tests", "datasets", "test_data")
        return SqliteDataset(base_dir)

    def test_initialization(self, sqlite_dataset):
        """Test that the dataset initializes correctly."""
        assert sqlite_dataset is not None
        assert sqlite_dataset.base_data_path == os.path.join("tests", "datasets", "test_data")
        dsets = sqlite_dataset.get_databases()
        assert len(dsets) == 1  # We expect only one database in the test data
        assert "test" in dsets  # The database name should be "test"

    def test_query_database(self, sqlite_dataset):
        """Test querying data from the test database."""
        # Query to get all customers from the customers table
        query = "SELECT customer_id, first_name, last_name FROM customers ORDER BY customer_id LIMIT 3"
        result = sqlite_dataset.query_database("test", query)

        # Check that we got the expected results
        assert result is not None
        assert len(result) == 3
        assert result[0]["customer_id"] == 1  # First customer ID should be 1
        assert result[0]["first_name"] == "John"  # First name should be John
        assert result[0]["last_name"] == "Doe"  # Last name should be Doe

    def test_query_database_join(self, sqlite_dataset):
        """Test a more complex query with a JOIN operation."""
        # Query to get order details with customer information
        query = """
        SELECT o.order_id, c.first_name, c.last_name, o.total_amount 
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.status = 'completed'
        ORDER BY o.order_id
        """
        result = sqlite_dataset.query_database("test", query)

        # Check that we got expected results
        assert result is not None
        assert len(result) == 2  # There should be 2 completed orders
        assert result[0]["order_id"] == 1  # First order ID
        assert result[0]["first_name"] == "John"  # Customer first name
        assert result[0]["total_amount"] == 929.98  # Total amount

    def test_query_database_aggregate(self, sqlite_dataset):
        """Test an aggregate query."""
        # Query to get the average price of Electronics products
        query = """
        SELECT AVG(price) as avg_price
        FROM products
        WHERE category = 'Electronics'
        """
        result = sqlite_dataset.query_database("test", query)

        # Check that we got expected results
        assert result is not None
        assert result[0]["avg_price"] is not None  # Average price should not be None
        assert abs(result[0]["avg_price"] - 576.66) < 0.1  # Average price of the 3 electronics items

    def test_nonexistent_database(self, sqlite_dataset):
        """Test behavior when trying to query a non-existent database."""
        with pytest.raises(Exception):  # Replace with the specific exception your class raises
            sqlite_dataset.query_database("nonexistent_db", "SELECT 1")

    def test_invalid_query(self, sqlite_dataset):
        """Test behavior when executing an invalid SQL query."""
        with pytest.raises(Exception):  # Replace with the specific exception your class raises
            sqlite_dataset.query_database("test", "SELECT * FROM nonexistent_table")

    def test_get_schema_dictionary(self, sqlite_dataset):
        """Test getting the schema dictionary for a specific database."""
        schema = sqlite_dataset.get_database_schema("test")
        assert type(schema) is dict
        assert "tables" in schema
        assert "customers" in schema["tables"]
        assert "orders" in schema["tables"]
        assert "products" in schema["tables"]
        assert "first_name" in schema["tables"]["customers"]["columns"]
        assert "total_amount" in schema["tables"]["orders"]["columns"]

    def test_describe_schema(self, sqlite_dataset):
        """Test describing the schema of a specific database."""
        supported_types = sqlite_dataset.get_schema_description_modes()
        assert type(supported_types) is list
        assert len(supported_types) > 0
        assert "basic" in supported_types
        assert "basic_types_relations" in supported_types
        assert "sql" in supported_types
        assert "datagrip" in supported_types
        for supported_type in supported_types:
            description = sqlite_dataset.describe_database_schema("test", mode=supported_type)
            assert type(description) is str
            assert len(description) > 0
            assert "first_name" in description
            assert "total_amount" in description
