import matplotlib.pyplot as plt
import dropbox
from pathlib import Path
from io import BytesIO
import subprocess

def upload(ax, project, path):
    bs = BytesIO()
    format = path.split('.')[-1]

    # Check if the file is a .tex file and handle it differently
    if format == 'tex':
        # Do nothing
        pass
    else:
        ax.savefig(bs, bbox_inches='tight', format=format)

    # token = os.DROPBOX
    token = subprocess.run(
        "curl https://api.dropbox.com/oauth2/token -d grant_type=refresh_token -d refresh_token=eztXuoP098wAAAAAAAAAAV4Ef4mnx_QpRaiqNX-9ijTuBKnX9LATsIZDPxLQu9Nh -u a415dzggdnkro3n:00ocfqin8hlcorr",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.split('{"access_token": "')[
        1].split('", "token_type":')[0]
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    if format == 'tex':
        # Handle .tex files by directly uploading their content
        dbx.files_upload(ax.encode(), f'/Apps/Overleaf/{project}/{path}', mode=dropbox.files.WriteMode.overwrite)
    else:
        dbx.files_upload(bs.getvalue(), f'/Apps/Overleaf/{project}/{path}', mode=dropbox.files.WriteMode.overwrite)


def plot_predicted(df, predicted, label, file, trainMethod, sDepVar, transformation='sum', show=False):
    # Plot the sum of predicted and actual sDepVar by date
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(df[df[trainMethod] == 0]['date'],
            df[df[trainMethod] == 0].groupby('date')[sDepVar].transform(transformation).astype(float), label='Actual',
            linestyle='dashed')
    ax.plot(df[df[trainMethod] == 0]['date'],
            df[df[trainMethod] == 0].groupby('date')[predicted].transform(transformation).astype(float), label=label)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Contribution (mDKK)')
    # ax.set_title('Out of Sample')
    ax.set_aspect('auto')
    ax.set_ylim([-5, 15.00])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
    plt.grid(alpha=0.5)
    plt.rcParams['axes.axisbelow'] = True
    plt.savefig(f"./Results/Figures/{file}.png")
    plt.savefig(f"./Results/Presentation/{file}.svg")
    upload(plt, 'Project-based Internship', f'figures/{file}.png')

    # Split file before the last underscore and add _1 to the end eg. 3_0_dst -> 3_0_1_dst
    file_fs = file.split('.')[0] + '_fs'

    # Plot the sum of predicted and actual sDepVar by date (full sample)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(df['date'],
            df.groupby('date')[sDepVar].transform(transformation).astype(float), label='Actual', linestyle='dashed')
    ax.plot(df['date'],
            df.groupby('date')[predicted].transform(transformation).astype(float), label=label)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Contribution (mDKK)')
    # ax.set_title('Full Sample')
    ax.set_aspect('auto')
    ax.set_ylim([-20, 100.00])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
    plt.grid(alpha=0.5)
    plt.rcParams['axes.axisbelow'] = True
    plt.savefig(f"./Results/Figures/FullSample/{file_fs}.png")
    plt.savefig(f"./Results/Presentation/FullSample/{file_fs}.svg")
    upload(plt, 'Project-based Internship', f'figures/{file_fs}.png')
    if show:
        plt.show()
    plt.close('all')
