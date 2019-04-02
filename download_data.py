import os
import sys

if not os.path.exists('cased_L-12_H-768_A-12'):
    try:
        import progressbar
        import urllib.request
        import zipfile
    except ImportError:
        sys.exit('Error importing modules')
    pbar = None
    def show_progress(block_num, block_size, total_size):
        global pbar
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()
        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None
    try:
        urllib.request.urlretrieve("https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip", "file.zip",show_progress)
        print('\nModel download complete')
    except urllib.error.HTTPError:
        sys.exit('Error in URL')

    with zipfile.ZipFile("file.zip","r") as zip_ref:
        zip_ref.extractall("cased_L-12_H-768_A-12")
    print('\nModel extracted to cased_L-12_H-768_A-12')
    
else:
    print('\nModel already downloaded and extracted to cased_L-12_H-768_A-12')