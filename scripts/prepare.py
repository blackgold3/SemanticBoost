from huggingface_hub import snapshot_download
def prepare():
    REPO_ID = 'Kleinhe/CAMD'
    snapshot_download(repo_id=REPO_ID, local_dir='./', local_dir_use_symlinks=False)

if __name__ == "__main__":
    prepare()