import numpy as np
import pandas as pd

import requests
import io
import urllib

from astropy.time import Time

# from etienne_extract_one_alert_radec import getztfmars


def select_gsa_events(time_min, time_max):
    '''

    Returns Gaia Science Alerts event names and coordinates from a selected range of time.

    :param time_min: Light curves generation start. A date, YYYY-MM-DD.
    :param time_max: Light curves generation end. A date, YYYY-MM-DD.
    :return: List of names and coordinates for which we will generate light curves.

    '''

    # download alerts from the GSA webpage
    url = r'http://gsaweb.ast.cam.ac.uk/alerts/alerts.csv'
    tables = pd.read_csv(url)  # Returns list of all tables on page
    # get names of the alerts
    names = tables["#Name"]
    # timestamps of when alert was published
    dates = tables[" Published"]
    # coordinates
    ra, dec = tables[" RaDeg"], tables[" DecDeg"]

    # Convert dates to JDs for easy comparison
    jd_min = Time(str(time_min).split(' ')[0]+'T00:00:00.0', scale='utc')
    jd_max = Time(str(time_max).split(' ')[0]+'T00:00:00.0', scale='utc')

    # Cycle through the GSA alerts. I will cut off sharply after time min is reached.
    selected_names, selected_ra, selected_dec = [], [], []
    for i in range(len(names)):
        pubdate = dates[i].split(' ')
        jd_published = Time(pubdate[0]+'T'+pubdate[1], scale='utc')
        if (jd_published >= jd_min and jd_published <= jd_max):
            selected_names.append(names[i]), selected_ra.append(ra[i]), selected_dec.append(dec[i])
    return selected_names, selected_ra, selected_dec


def get_lightcurve_ZTF_alert(ra, dec):
    mars_time, mars_mag, mars_err, mars_filter = getztfmars(ra, dec)
    marsmjd_g = []
    marsmag_g = []
    marserr_g = []
    marsmjd_r = []
    marsmag_r = []
    marserr_r = []
    fol = 0.

    # print mars_filter
    if (len(mars_time) > 1):
        fol = 1
        for i in range(len(mars_time)):
            if (mars_filter[i] == 'g'):
                # print mars_filter[i] , mars_mag[i]
                if (mars_mag[i] is not None):
                    marsmjd_g.append(float(mars_time[i]) - 2450000.5)
                    marsmag_g.append(float(mars_mag[i]))
                    marserr_g.append(float(mars_err[i]))
                # print "Elo!"
                # if (printdata==1): print("{} {} {} {}<br>".format(float(mars_time[i])-2400000.5, float(mars_mag[i]), float(mars_err[i]), mars_filter[i]))
            if (mars_filter[i] == 'r'):
                # print mars_filter[i], mars_mag[i]
                if (mars_mag[i] is not None):
                    marsmjd_r.append(float(mars_time[i]) - 2450000.5)
                    marsmag_r.append(float(mars_mag[i]))
                    marserr_r.append(float(mars_err[i]))
                # print "Elo2!"
                # if (printdata==1): print("{} {} {} {}<br>".format(float(mars_time[i])-2400000.5, float(mars_mag[i]), float(mars_err[i]), mars_filter[i]))
    marsmjd_g = np.asarray(marsmjd_g)
    marsmag_g = np.asarray(marsmag_g)
    marserrd_g = np.asarray(marserr_g)
    marsmjd_r = np.asarray(marsmjd_r)
    marsmag_r = np.asarray(marsmag_r)
    marserrd_r = np.asarray(marserr_r)

    # print mars_time, marsmjd_g, marsmjd_r
    ztf_data = []
    ztf_bands = []
    ztf_data.append((marsmjd_g, marsmag_g, marserr_g))
    ztf_bands.append("ZTF_al_g")

    ztf_data.append((marsmjd_r, marsmag_r, marserr_r))
    ztf_bands.append("ZTF_al_r")

    if (fol > 0):
        print("Data from ZTF Alerts obtained!")
    else:
        print("No ZTF Alerts data available.")

    return fol, ztf_data, ztf_bands


def get_lightcurve_ZTF_DR(ra, dec):
    rad = 0.5 / 3600.
    bands = ['g', 'r', 'i']

    datasets = []
    b = []
    try:
        for band in bands:
            data = []
            url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+%f+%f+%f&BANDNAME=%s&NOBS_MIN=3&BAD_CATFLAGS_MASK=32768&FORMAT=ipac_table" % (
            ra, dec, rad, band)
            print(url)
            req = requests.get(url)
            text = req.text
            lines = text.split('\n')

            i = 0
            for line in lines:
                col = line.split()
                # print(len(col), col)
                if (len(col) > 1):
                    if (col[0][0] != "|" and col[0][0] != "\\" and col[0][0] != "<"):
                        # print(i, col[2], col[4], col[5], col[7])
                        if (float(col[5]) > 1e-8):
                            data.append((float(col[2]), float(col[4]), float(col[5])))
                i += 1
            data = np.asarray(data)
            if (len(data) > 0):
                datasets.append(data)
                b.append(band)


    except requests.HTTPError as exception:
        print(exception)

    return datasets, b


def get_lightcurve_gaia(name):
    url = ("http://gsaweb.ast.cam.ac.uk/alerts/alert/%s/lightcurve.csv") % (name)

    try:
        req = requests.get(url)
        text = req.text
        lines = text.split('\n')
        times = []
        mags = []

        for l in lines:
            col = l.split(',')
            # print(col)
            if (len(col) > 1):
                if (len(col) == 3 and (col[1] != 'JD(TCB)')):
                    if (col[2] != 'null' and col[2] != 'untrusted'):
                        times.append(float(col[1]))
                        mags.append(float(col[2]))
        return np.array(times), np.array(mags)
    except requests.HTTPError as exception:
        print(exception)
        return 0,0

def get_gaia_errors(mag):
    a1 = 0.2
    b1 = -5.2
    a2 = 0.26
    b2 = -6.26

    err = []
    for i in range(0, len(mag)):
        if mag[i] <= 13.5:
            err_corr = a1 * 13.5 + b1
        elif mag[i] > 13.5 and mag[i] <= 17.:
            err_corr = a1 * mag[i] + b1
        elif mag[i] > 17.:
            err_corr = a2 * mag[i] + b2
        err.append(10. ** err_corr)
    return err


def get_lightcurve_MOA(name, field):
    text = name.split('-')
    # print(text)
    year = text[1]

    if (int(year) <= 2015):
        url = "http://www.massey.ac.nz/~iabond/moa/alerts/view_txt.php?url=http://it047333.massey.ac.nz/moa/ephot/phot-%s.dat" % (
            field)
    else:
        url = ("https://www.massey.ac.nz/~iabond/moa/alert%s/fetchtxt.php?path=moa/ephot/phot-%s.dat") % (year, field)
    # print(url)
    times = []
    mags = []
    errs = []

    try:
        req = requests.get(url)
        text = req.text
        lines = text.split('\n')
        # print(len(lines))

        i = 0
        if (len(lines) > 10):
            for l in lines:
                # print(i)
                # print(l)
                if (len(l) > 0):
                    col = l.split()
                    if (i > 10 and i < len(lines) - 2 and float(col[1]) > 0.):
                        # print(l)
                        times.append(float(col[0]))
                        mags.append(float(col[1]))
                        errs.append(float(col[2]))
                i += 1
    except requests.HTTPError as exception:
        print(exception)

    return np.array(times), np.array(mags), np.array(errs)


def get_lightcurve_OGLE_EWS(name):
    text = name.split('-')
    # print(text)
    year = text[1]
    num = text[3]

    if(text[2] == "BLG"):
        url = ("http://www.astrouw.edu.pl/ogle/ogle4/ews/%s/blg-%s/phot.dat") % (year, num)
    elif(text[2] == "GD"):
        url = ("http://www.astrouw.edu.pl/ogle/ogle4/ews/%s/gd-%s/phot.dat") % (year, num)
    else:
        url = ("http://www.astrouw.edu.pl/ogle/ogle4/ews/%s/dg-%s/phot.dat") % (year, num)

    try:
        req = requests.get(url).content
        ogleLc = pd.read_csv(io.StringIO(req.decode('utf-8')), delimiter=" ", header=None)
        # print(ogleLc)
        data = ogleLc.to_numpy()
        return data[:, 0], data[:, 1], data[:, 2]
    requests.HTTPError as exception:
        print(exception)
        return 0,0,0
