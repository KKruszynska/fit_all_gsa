import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cycler

import urllib, base64
import requests
import io
import os

from pyLIMA.outputs.pyLIMA_plots import create_telescopes_to_plot_model
from pyLIMA import toolbox as ptool
import  pyLIMA.fits.objective_functions as pfof

def plot_data(name, datasets, n_telescopes, tel_labels, tmin, tmax):
    fig = plt.figure()
    plt.grid(color='0.95')
    plt.title(name)
    plt.gca().invert_yaxis()
    color = plt.cm.jet(np.linspace(0.01, 0.99, n_telescopes))  # This returns RGBA; convert:
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255), 'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler(color=hexcolor)

    if(n_telescopes == 1):
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        # print(type(datasets))
        # print(len(datasets))
        # print("----------------------------------------------")
        plt.errorbar(datasets[0,:] - 2450000., datasets[1,:], yerr=datasets[2,:], color=color, marker='o',
                     label=tel_labels, linestyle='')
    else:
        i = 0
        for data in datasets:
            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
            plt.errorbar(data[0, :] - 2450000., data[1, :], yerr=data[2, :], color=color, marker='o',
                         label=tel_labels[i], linestyle='')
            i += 1

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
def plot_lightcurve(name, microlensing_model, model_parameters, tmin, tmax):
    # ------------------- Same as in pyLIMA ------------------------------------
    # Change matplotlib default colors
    n_telescopes = len(microlensing_model.event.telescopes)
    color = plt.cm.jet(np.linspace(0.01, 0.99, n_telescopes))  # This returns RGBA; convert:
    hexcolor = ['#' + format(int(i[0] * 255), 'x').zfill(2) + format(int(i[1] * 255), 'x').zfill(2) +
                format(int(i[2] * 255), 'x').zfill(2) for i in color]
    matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler(color=hexcolor)
    # -------------------------------------------------------------------------
    fig = plt.figure()
    plt.grid(color='0.95')
    grid = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axes = plt.subplot(grid[0])
    plt.gca().invert_yaxis()
    plt.title(name)
    plt.ylabel("magnitude")
    plt.xlim(tmin - 2450000., tmax - 2450000.)
    # plot data
    # ------------------- Same as in pyLIMA ------------------------------------
    pyLIMA_parameters = microlensing_model.compute_pyLIMA_parameters(model_parameters)
    list_of_telescopes = create_telescopes_to_plot_model(microlensing_model, pyLIMA_parameters)
    telescopes_names = np.array([i.name for i in microlensing_model.event.telescopes])

    index = 0

    ref_names = []
    ref_locations = []
    ref_magnification = []
    ref_fluxes = []

    for ref_tel in list_of_telescopes:
        model_magnification = microlensing_model.model_magnification(ref_tel, pyLIMA_parameters)
        microlensing_model.derive_telescope_flux(ref_tel, pyLIMA_parameters, model_magnification)

        f_source = getattr(pyLIMA_parameters, 'fsource_' + ref_tel.name)
        f_blend = getattr(pyLIMA_parameters, 'fblend_' + ref_tel.name)

        ref_names.append(ref_tel.name)
        ref_locations.append(ref_tel.location)
        ref_magnification.append(model_magnification)
        ref_fluxes.append([f_source, f_blend])

    for ind, tel in enumerate(microlensing_model.event.telescopes):
        if tel.lightcurve_flux is not None:

            if tel.location == 'Earth':
                ref_index = np.where(np.array(ref_locations) == 'Earth')[0][0]

            else:
                ref_index = np.where(np.array(ref_names) == tel.name)[0][0]

            residus_in_mag = pfof.photometric_residuals_in_magnitude(
                    tel, microlensing_model,
                    pyLIMA_parameters)
            if ind == 0:
                reference_source = ref_fluxes[ind][0]
                reference_blend = ref_fluxes[ind][1]
                index += 1

            time_mask = []
            for time in tel.lightcurve_flux['time'].value:
                time_index = np.where(list_of_telescopes[ref_index].lightcurve_flux[
                                          'time'].value == time)[0][0]
                time_mask.append(time_index)

            model_flux = reference_source * ref_magnification[ref_index][
                time_mask] + reference_blend
            magnitude = ptool.brightness_transformation.ZERO_POINT - 2.5 * \
                        np.log10(model_flux)

            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][ind]
            marker = str(MARKER_SYMBOLS[0][ind])
            # ----------------------------------------------------------------------------------------
            axes.errorbar(tel.lightcurve_magnitude['time'].value,
                          magnitude + residus_in_mag,
                          tel.lightcurve_magnitude['err_mag'].value,
                          color=color, marker=marker,
                          label=tel.name, linestyle='')

    # plot model
    # ------------------- Same as in pyLIMA ------------------------------------
    tel = list_of_telescopes[0]
    ref_source = getattr(pyLIMA_parameters, 'fsource_' + tel.name)
    ref_blend = getattr(pyLIMA_parameters, 'fblend_' + tel.name)

    magni = microlensing_model.model_magnification(tel, pyLIMA_parameters)
    microlensing_model.derive_telescope_flux(tel, pyLIMA_parameters, magni)

    magnitude = ptool.brightness_transformation.ZERO_POINT - 2.5 * np.log10(ref_source * magni + ref_blend)
    index_color = np.where(tel.name == telescopes_names)[0][0]
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][index_color]
    # ------------------------------------------------------------------------------
    axes.plot(tel.lightcurve_magnitude['time'].value, magnitude, c=color, label=name, linestyle='-')

    plt.grid(True)
    plt.legend(loc='best')

    axes = plt.subplot(grid[1])
    # plotting residuals
    # ------------------- Same as in pyLIMA ------------------------------------
    for ind, tel in enumerate(microlensing_model.event.telescopes):

        if tel.lightcurve_flux is not None:
            residus_in_mag = \
                pyLIMA.fits.objective_functions.photometric_residuals_in_magnitude(
                    tel, microlensing_model, pyLIMA_parameters)

            color = plt.rcParams["axes.prop_cycle"].by_key()["color"][ind]
            marker = str(MARKER_SYMBOLS[0][ind])

            plots.plot_light_curve_magnitude(tel.lightcurve_magnitude['time'].value,
                                             residus_in_mag,
                                             tel.lightcurve_magnitude['err_mag'].value,
                                             figure_axe=figure_axe, color=color,
                                             marker=marker, name=tel.name)

    # ---------------------------------------------------------------------------------
            axes.errorbar(tel.lightcurve_magnitude['time'].value,
                  residus_in_mag,
                  tel.lightcurve_magnitude['err_mag'].value,
                  color=color, marker=marker, linestyle='')

    plt.ylabel("res")
    plt.xlabel("JD-2450000.")
    plt.xlim(tmin - 2450000., tmax - 2450000.)


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