# drop rows non-english
import pandas as pd
from langdetect import detect
file_name = "needs_tweets"
file_name_input = file_name + '.csv'
file_name_output = file_name + '_eng.csv'
file_delimiter_input = ','
file_delimiter_output = ','
df = pd.read_csv(file_name_input, delimiter=file_delimiter_input, encoding = "ISO-8859-1")
print('shape before', df.shape)
for j, row in df.iterrows():
    text = row['tweet_text']
    # print(j, '-----', text)
    # s = ''
    # s = ''.join(i for i in text if ord(i)>128)
    found = False


    # #for testing 20190403, keep other languages
    # for i in text:
    #     #from langdetect import detect
    #
    #
    #     if ord(i)>128:
    #         found = True
    #         break
    #
    #
    # if (found):
    #     df.drop(j, inplace=True)

    #20190404
    if detect(text) != 'en':
        # print('deleting row: ', j, '-----', text)
        df.drop(j, inplace=True)


# df['tweet_text'] = df.apply(lambda row: detect(row['tweet_text'].decode("utf8")), axis=1)


df.to_csv(file_name_output, sep=file_delimiter_output, encoding='utf-8', index=False)
# for j, row in df.iterrows():
#     if not wordnet.synsets(df.i[j]):#Comparing if word is non-English
#            df.drop(j, inplace=True)

print('shape after', df.shape)
