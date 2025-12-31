import os
import argparse
import pandas as pd
from .classes import RathCelonDam


def process_excel_input(excel_file):
    """Process input from an Excel file, treating each row as a dam configuration."""
    print(f'Opening Excel database: {excel_file}')
    df = pd.read_excel(excel_file)

    # Convert DataFrame to list of dictionaries (one per dam)
    dams = df.to_dict(orient='records')

    for dam_row in dams:
        dam_name = dam_row.get("name", "Unknown Dam")
        output_dir = dam_row.get("output_dir")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Processing dam: {dam_name}")
        # Pass the entire row dictionary. The class __init__ will
        # now pick the correct flowline/streamflow columns.
        dam_instance = RathCelonDam(dam_row=dam_row)
        dam_instance.process_dam()

def main():
    parser = argparse.ArgumentParser(description="Process rating curves from an Excel LHD database.")
    parser.add_argument("excel_file", type=str, help="Path to the Excel file")
    args = parser.parse_args()

    process_excel_input(args.excel_file)

if __name__ == "__main__":
    main()