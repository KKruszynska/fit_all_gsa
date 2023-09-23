import numpy as np
import matplotlib.pyplot as plt

import urllib, base64
import requests
import io

from pyLIMA.outputs import pyLIMA_plots
from cycler import cycler
from pyLIMA import toolbox as ptool

from datetime import datetime
from astropy.time import Time

def color_marker_dicts():
    name_telescope = ['ASASSN_g', 'LCO_1m_gp', 'ZTF_al_g', 'ZTF_g',
                      'ASASSN_V',
                      'Gaia', 'LCO_1m_rp', 'ZTF_al_r', 'ZTF_r',
                      'KMTNet', 'LCO_1m_ip', 'MOA', 'OGLE', 'ZTF_al_i', 'ZTF_i',
                      'Danish_Lucky_Z'
                      ]
    n_telescopes = len(name_telescope)

    # colors
    color = plt.cm.jet(np.linspace(0.01, 0.99, n_telescopes))  # This returns RGBA; convert:
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255), 'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]
    # markers
    MARKER_SYMBOLS = np.array(
        [['o', '.', '*', 'v', '^', '<', '>', 's', 'p', 'd', 'x'] * 10])
    marker_cycle = MARKER_SYMBOLS[0][:n_telescopes]

    color_dict = dict(zip(name_telescope, hexcolor))
    marker_dict = dict(zip(name_telescope, marker_cycle))

    return color_dict, marker_dict

def plot_data(name, datasets, n_telescopes, tel_labels, tmin, tmax):
    color_dict, marker_dict = color_marker_dicts()
    custom_color, custom_marker = [], []
    for lab in tel_labels:
        custom_color.append(color_dict[lab])
        custom_marker.append(marker_dict[lab])

    fig = plt.figure()
    plt.grid(color='0.95')
    plt.title(name)
    plt.gca().invert_yaxis()

    if(n_telescopes == 1):
        color = custom_color[0]
        marker = custom_marker[0]
        # print(type(datasets))
        # print(len(datasets))
        # print("----------------------------------------------")
        plt.errorbar(datasets[0,:] - 2450000., datasets[1,:], yerr=datasets[2,:], color=color, marker=marker,
                     label=tel_labels, linestyle='')
    else:
        i = 0
        for data in datasets:
            color = custom_color[0]
            marker = custom_marker[0]
            plt.errorbar(data[0, :] - 2450000., data[1, :], yerr=data[2, :], color=color, marker=marker,
                         label=tel_labels[i], linestyle='')
            i += 1

    today = Time.now()
    jd_today = today.jd
    plt.axvline(x=jd_today, label='Now', ls='--', color='darkslategray')

    plt.ylabel("magnitude")
    plt.xlabel("JD-2450000")
    plt.xlim(tmin, tmax)
    plt.legend(loc='best')

    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode('utf8')
    html_code = "data:image/png;base64," + encoded
    img.close()
    plt.close(fig)

    return html_code

# I had to modify original pyLIMA plotting functions to get the desired effect
# Original code by Bachelet et al.
def plot_lightcurve(name, fit, model, tmin, tmax):
    # ------------------- Custom colors  ------------------------------------
    color_dict, marker_dict = color_marker_dicts()
    custom_color, custom_marker = [], []
    for tel in model.event.telescopes:
        custom_color.append(color_dict[tel.name])
        custom_marker.append(marker_dict[tel.name])
    # -------------------------------------------------------------------------

    ### Find the telescope fluxes if needed

    if len(fit.fit_results['best_model']) != len(model.model_dictionnary):
        telescopes_fluxes = model.find_telescopes_fluxes(fit.fit_results['best_model'])
        telescopes_fluxes = [getattr(telescopes_fluxes, key) for key in
                             telescopes_fluxes._fields]

        model_parameters1 = np.r_[fit.fit_results['best_model'], telescopes_fluxes]

    else:

        model_parameters1 = fit.fit_results['best_model']

    # custom colors and markers
    custom_cycler = (cycler(color=custom_color))

    pyLIMA_plots.MARKERS_COLORS = custom_cycler
    pyLIMA_plots.MARKER_SYMBOLS = np.array([custom_marker])
    # ------------------- Plotting  ------------------------------------
    fig, axes = plt.subplots(2, 1, height_ratios=[3, 1])
    # ------------------- Plotting lc ------------------------------------
    axes[0].title.set_text(name)
    axes[0].set_ylabel("magnitude")
    axes[0].grid(True, color='0.95')
    axes[0].invert_yaxis()

    axes[0].set_xlim(tmin, tmax)

    # Plot model1 and align data to it
    pyLIMA_plots.plot_photometric_models(axes[0], model, model_parameters1, plot_unit='Mag')
    pyLIMA_plots.plot_aligned_data(axes[0], model, model_parameters1, plot_unit='Mag')

    # Plot today's date
    today = Time.now()
    jd_today = today.jd
    axes[0].axvline(x=jd_today, label='Now', ls='--', color='darkslategray')


    # plot model
    # ------------------- Plotting residuals  ------------------------------------
    axes[1].set_ylabel("Res")
    axes[1].grid(True, color='0.95')
    axes[1].invert_yaxis()
    axes[1].set_xlim(tmin, tmax)
    axes[1].set_ylim(-0.5, 0.5)

    pyLIMA_plots.plot_residuals(axes[1], model, model_parameters1, plot_unit='Mag')

    axes[1].axhline(y=0, color='black', ls='-')
    axes[1].axvline(x=jd_today, label='Now', ls='--', color='darkslategray')

    axes[0].legend(shadow=True, fontsize='large',
                   bbox_to_anchor=(0, 1.02, 1, 0.2),
                   loc="lower left",
                   mode="expand", borderaxespad=0, ncol=3)


    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode('utf8')
    html_code = "data:image/png;base64,"+encoded
    img.close()
    plt.close(fig)

    return html_code

def plot_sdss(ra,dec,sizex=150, sizey=150, scale=0.27):
 try:
      #urllib.urlopen("http://skyservice.pha.jhu.edu")
    #url = ("http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx?ra=%s&dec=%s&scale=%f&width=%d&height=%d&opt=GST&query=SR(10,20)")%(ra,dec,scale,sizex,sizey)
    #        im='data:image/jpeg;base64,' + (base64.b64encode(requests.get(url).content))
    url="http://skyserver.sdss.org/dr13/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Chart.Image&ra=%f&dec=%f&width=%d&height=%d&scale=%f&opt=GST&query=SR(10,20)"%(ra,dec,sizex,sizey,scale)
    response = requests.get(url)
    im = ("data:" + response.headers['Content-Type'] + ";" + "base64," + urllib.parse.quote(base64.b64encode(response.content)))
    return im
 except IOError:
    return 0

def plot_xmatchps1jpeg(ra, dec):

 image_url = None
 try:
     urllib.request.urlopen("http://ps1images.stsci.edu/")
     urlfnames = urllib.request.urlopen(
         'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra=%.6f&dec=%.6f' % ((ra), (dec)))
     fnames = np.genfromtxt(urlfnames.read().split('\n'), names=True, dtype=None)
     if fnames.size == 0 or not np.any(fnames['filter'] == 'y') or not np.any(
             fnames['filter'] == 'i') or not np.any(fnames['filter'] == 'g'):
         image_url = None
     else:
         yname = fnames[fnames['filter'] == 'y']['filename'].item(0)
         iname = fnames[fnames['filter'] == 'i']['filename'].item(0)
         gname = fnames[fnames['filter'] == 'g']['filename'].item(0)
         image_url = 'http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?format=png&size=144&zoom=1.5&ra=%.6f&dec=%.6f' % (
         ra, dec) + '&red=' + yname + '&green=' + iname + '&blue=' + gname
         urllib.request.urlopen(image_url)  # openning the url to check if all is fine
 except IOError:
     image_url = None

 if image_url != None:
     if urllib.request.urlopen(image_url).getcode() == 500:
         image_url = None

 if image_url == None:
     fig = plt.figure(figsize=(1.5, 1.5))
     frame = fig.add_axes([0., 0., 1., 1.])
     plt.text(0, 0.65, 'PS1 archive not')
     plt.text(0, 0.5, 'available')

     img = io.BytesIO()
     fig.savefig(img, format='png',
                 bbox_inches='tight')
     img.seek(0)
     encoded = base64.b64encode(img.getvalue())
     html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
     img.close()
     plt.close(fig)

     return html_code

 else:

     im = plt.imread(image_url)
     fig = plt.figure(figsize=(1.5, 1.5))
     ax1 = plt.Axes(fig, [0., 0., 1., 1.])
     ax1.set_axis_off()
     fig.add_axes(ax1)
     plt.imshow(im)
     plt.scatter(len(im) / 2 - 0.5, len(im) / 2 - 0.5, marker='+', c='k')

     img = io.BytesIO()
     fig.savefig(img, format='png',
                 bbox_inches='tight')
     img.seek(0)
     encoded = base64.b64encode(img.getvalue())
     html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
     img.close()
     plt.close(fig)

     return html_code