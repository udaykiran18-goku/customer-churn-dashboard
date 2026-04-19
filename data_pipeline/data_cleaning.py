import pandas as pd


def clean_data(data):
    print("\nStarting Data Cleaning...")

    print("Original Shape:", data.shape)

    # Remove missing customer IDs
    data = data.dropna(subset=["CustomerID"])

    # Remove negative or zero quantities
    data = data[data["Quantity"] > 0]

    # Remove negative prices
    data = data[data["UnitPrice"] > 0]

    # Remove cancelled orders (InvoiceNo starting with 'C')
    data = data[~data["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove duplicates
    data = data.drop_duplicates()

    print("Cleaned Shape:", data.shape)

    return data