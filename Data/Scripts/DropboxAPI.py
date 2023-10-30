import dropbox
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt


def upload(ax, project, path):
    bs = BytesIO()
    format = path.split('.')[-1]
    ax.figure.savefig(bs, bbox_inches='tight', format=format)

    token = Path('Data/.AUX/Dropbox.txt').read_text()
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    dbx.files_upload(
        f=bs.getvalue(),
        path=f'/Apps/Overleaf/{project}/{path}',
        mode=dropbox.files.WriteMode.overwrite)
