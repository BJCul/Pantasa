import gzip
 
chunk_size = 2
chunk_size_remove
with gzip.open('tl_dedup.txt.gz', 'rb') as f_in:
  with open('dataset.txt', 'wb') as f_out:
    chunk = f_in.read(chunk_size)
    while chunk:
      f_out.write(chunk)
      chunk = f_in.read(chunk_size) 