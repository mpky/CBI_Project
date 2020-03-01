import numpy as np
import pandas as pd

def clean_first_range(output_csv=False):
    raw_data1 = pd.read_csv('./data/raw/124_394.csv')

    # chop off last row
    raw_data1 = raw_data1.iloc[:259,]

    # strip new lines
    raw_data1['total_for_foreign'] = raw_data1['total_for_foreign'].str.strip()
    raw_data1['total_cash'] = raw_data1['total_cash'].str.strip()
    raw_data1['grand_total'] = raw_data1['grand_total'].str.strip()

    # Remove commas and convert to int
    raw_data1['total_for_foreign'] = raw_data1.total_for_foreign.str.replace(',','')
    raw_data1['total_for_foreign'] = pd.to_numeric(raw_data1.total_for_foreign,errors='coerce')
    raw_data1['total_cash'] = pd.to_numeric(raw_data1.total_cash.str.replace(',',''), errors='coerce')
    raw_data1['grand_total'] = pd.to_numeric(raw_data1.grand_total.str.replace(',',''), errors='coerce')

    if output_csv:
        # output CSV with cleaned columns
        return raw_data1.to_csv(output_path,index=False)
    return raw_data1

def clean_second_range(output_csv=False):
    raw_data2 = pd.read_csv('./data/raw/394_749.csv')

    raw_data2['total_for_foreign'] = raw_data2['total_for_foreign'].str.strip()
    raw_data2['total_cash'] = raw_data2['total_cash'].str.strip()
    raw_data2['grand_total'] = raw_data2['grand_total'].str.strip()

    # now remove commas and convert to float
    raw_data2['total_for_foreign'] = pd.to_numeric(raw_data2.total_for_foreign.str.replace(',',''), errors='coerce')
    raw_data2['total_cash'] = pd.to_numeric(raw_data2.total_cash.str.replace(',',''), errors='coerce')
    raw_data2['grand_total'] = pd.to_numeric(raw_data2.grand_total.str.replace(',',''), errors='coerce')

    if output_csv:
        # save to csv
        return raw_data2.to_csv(output_path,index=False)
    return raw_data2

# Combine both dfs
def combine_data():
    range1_df = clean_first_range()
    range2_df = clean_second_range()

    combined_df = range1_df.append(range2_df,ignore_index=True)
    combined_df.to_csv('./data/processed/processed.csv',index=False)
    # return combined_df

if __name__ == '__main__':
    combine_data()
