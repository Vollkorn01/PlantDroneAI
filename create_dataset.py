import pandas as pd
import urllib.request

# import txt file of dataset with image urls (obtained from
# https://www.gbif.org/occurrence/download/0018710-190621201848488)
data = pd.read_csv('multimedia_other.txt', sep='\t', error_bad_lines=False)

# download images and save in folder
for index, row in data.iterrows():
    try:
        print(index, row['identifier'])
        urllib.request.urlretrieve(row['identifier'], './images_other/' + str(index) + '_grass.jpg')
    except:
        print('failed, next')