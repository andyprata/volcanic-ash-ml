import pandas as pd


url = 'http://ds.data.jma.go.jp/svd/vaac/data/Archives/2021_vaac_list.html'
table = pd.read_html(url)
df = table[1]
# Drop last header in array because it's 'Unnamed'
new_headers = df.keys().values[:-1]
# Drop the column with repeated datetime string
df = df.drop(['Volcano'], axis=1)
# Headers will now be wrong so we need to rename them
old_headers = df.keys().values
for old, new in zip(old_headers, new_headers):
    df = df.rename(columns={old: new})
# Print dataframe
print(df.head())

# Now.. it looks like the file name is YYYYmmdd_volcanoid_advisorynumber_Text.html. We have the datetimes and advisory numbers in the table above.
# So it would just be a matter of converting volcano name to volcanoid. E.g. if 'KARYMSKY': volcanoid = 30013000 and then wget-ing the list.