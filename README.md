# SmartPhoneSentiment
Use AWS EMR to collect the web data in the web database compiled by Common Crawl.
Helpful webpages are those that contain a smart phone review and positive/negative sentiment.
Mapper script scans teh Common Crawl for a smart phone name and the relevant words.  This script sends information back to the reducer.
Reducer script accumulates the analysis from the individual mapper jobs and writes it to the output file on S3.
Aggregation script helps stitch together the raw output from the multiple job flows.
