import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
import astropy.units as u

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

def ogle_xmatch(names, ra, dec):
    """
    Returns a list of Gaia alert names and corresponding OGLE EWS alerts.

    Args:
        - names string - name of Gaia alert
        - ra, dec   float - coordinates on the sky in equatorial frame, degrees
    Returns:
        - xmatch_result list - Gaia alert name (first element) and corresponding OGLE EWS alert name (second element)
    """
    rad = 1.0 / 60. / 60.  # 0.5 arcsec search radius
    xmatch_result = []
    ogle_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2022, 2023, 2024]
    for year in ogle_years:
        url = r'https://www.astrouw.edu.pl/ogle/ogle4/ews/%d/lenses.par' % year
        tables = pd.read_csv(url, header=0, delim_whitespace=True)
        # print(tables)
        catalog = SkyCoord(ra=tables["RA(J2000)"].values[:], dec=tables["Dec(J2000)"].values[:],
                           unit=(u.hourangle, u.deg))

        c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
        idxc, idxcatalog, d2d, d3d = catalog.search_around_sky(c, rad * u.deg)
        for i in range(len(idxc)):
            xmatch_result.append((names[idxc[i]], "OGLE-%s"%tables["Event"].values[idxcatalog[i]]))

    return xmatch_result


def kmtn_xmatch(names, ra, dec):
    """
    Returns a list of Gaia alert names and corresponding KMTNet alerts.

    Args:
        - names string - name of Gaia alert
        - ra, dec   float - coordinates on the sky in equatorial frame, degrees
    Returns:
        - xmatch_result list - Gaia alert name (first element) and corresponding KMTNet alert name (second element)
    """
    rad = 1.0 / 60. / 60.  # 0.5 arcsec search radius
    xmatch_result = []
    kmtn_years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    for year in kmtn_years:
        url = r'https://kmtnet.kasi.re.kr/~ulens/event/%d/' % (year)
        tables = pd.read_html(url, header=0)
        # print(tables[0].columns.values.tolist())
        catalog = SkyCoord(ra=tables[0]["RA"].values[:], dec=tables[0]["Dec"].values[:], unit=(u.hourangle, u.deg))

        c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
        idxc, idxcatalog, d2d, d3d = catalog.search_around_sky(c, rad * u.deg)
        for i in range(len(idxc)):
            xmatch_result.append((names[idxc[i]], tables[0]["Event"].values[idxcatalog[i]]))

    return xmatch_result


def moa_xmatch(names, ra, dec):
    """
    Returns a list of Gaia alert names and corresponding MOA alerts.
    Warning! MOA alerts webpages for 2013-2015 dont work anymore

    Args:
        - names string - name of Gaia alert
        - ra, dec   float - coordinates on the sky in equatorial frame, degrees
    Returns:
        - xmatch_result list - Gaia alert name (first element), corresponding MOA alert name (second element)
                               and MOA field (third element)
    """

    rad = 1.0 / 60. / 60.  # 0.5 arcsec search radius
    xmatch_result = []

    c = SkyCoord(ra=ra * u.degree,
                 dec=dec * u.degree)

    # for year in [2013, 2014, 2015]:
    #     url = r'http://www.massey.ac.nz/~iabond/moa/alerts/listevents.php?year=%d' % (year)
    #     tables = pd.read_html(url)  # Returns list of all tables on page
    #     nan_value = float("NaN") # Nan value to convert from empty string
    #     tables[0].replace("", nan_value, inplace=True)
    #     tables[0].columns = ["Name", "Field", "RA", "Dec"]
    #     alert_table = tables[0].dropna(subset=["RA", "Dec"])
    #
    #     catalog = SkyCoord(ra=alert_table.values[:, 2],
    #                        dec=alert_table.values[:, 3], unit=(u.hourangle, u.deg))
    #
    #     idxc, idxcatalog, d2d, d3d = catalog.search_around_sky(c, rad * u.deg)
    #
    #     for i in range(len(idxc)):
    #         xmatch_result.append(
    #             (names[idxc[i]], "MOA-%s"%alert_table["Name"].values[idxcatalog[i]], alert_table["Field"].values[idxcatalog[i]]))

    # for year in [2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    # url = r'http://www.massey.ac.nz/~iabond/moa/alert%d/index.dat' % (year)
    url = r'https://moaprime.massey.ac.nz/alerts/index/moa/2025'
    # tables = pd.read_csv(url, delimiter=" ", header=None)  # Returns list of all tables on page
    tables = pd.read_html(url) # Returns list of all tables on page
    nan_value = float("NaN")  # Nan value to convert from empty string
    tables[0].replace("", nan_value, inplace=True)
    tables[0].columns = ["Name", "Alert_Date", "RA", "Dec", "t0", "te", "Amax", "I0"]
    alert_table = tables[0].dropna(subset=["RA", "Dec"])
    catalog = SkyCoord(ra=alert_table["RA"].values[:], dec=alert_table["Dec"].values[:], unit=(u.hourangle, u.deg))

    idxc, idxcatalog, d2d, d3d = catalog.search_around_sky(c, rad * u.degree)

    for i in range(len(idxc)):
        xmatch_result.append(
            (names[idxc[i]], "MOA-%s"%alert_table["Name"].values[idxcatalog[i]]))

    # Prime Alerts 2024
    url = r'https://moaprime.massey.ac.nz/prime2024'
    tables = pd.read_html(url)  # Returns list of all tables on page
    nan_value = float("NaN")  # Nan value to convert from empty string
    tables[0].replace("", nan_value, inplace=True)
    tables[0].columns = ["Name", "InternalID", "RA", "Dec", "t0", "te", "Amax"]
    alert_table = tables[0].dropna(subset=["RA", "Dec"])
    catalog = SkyCoord(ra=alert_table["RA"].values[:], dec=alert_table["Dec"].values[:], unit=(u.hourangle, u.deg))

    idxc, idxcatalog, d2d, d3d = catalog.search_around_sky(c, rad * u.degree)

    for i in range(len(idxc)):
        xmatch_result.append(
            (names[idxc[i]], "PRIME-%s" % alert_table["Name"].values[idxcatalog[i]]))

    # PRIME Alerts 2025
    url = r'https://moaprime.massey.ac.nz/alerts/index/prime/2025'
    tables = pd.read_html(url)  # Returns list of all tables on page
    nan_value = float("NaN")  # Nan value to convert from empty string
    tables[0].replace("", nan_value, inplace=True)
    tables[0].columns = ["Name", "AlertDate", "RA", "Dec", "t0", "te", "Amax", "H0"]
    alert_table = tables[0].dropna(subset=["RA", "Dec"])
    catalog = SkyCoord(ra=alert_table["RA"].values[:], dec=alert_table["Dec"].values[:], unit=(u.hourangle, u.deg))

    idxc, idxcatalog, d2d, d3d = catalog.search_around_sky(c, rad * u.degree)

    for i in range(len(idxc)):
        xmatch_result.append(
            (names[idxc[i]], "PRIME-%s" % alert_table["Name"].values[idxcatalog[i]]))

    # Oby osobę, która jest za to odpowiedzialna, ścisnęły drzwi >:[
    # year = 2017
    #
    # url = r'http://www.massey.ac.nz/~iabond/moa/alert%d/index.dat' % (year)
    # tables1 = pd.read_csv(url, delimiter=" ", header=None)  # Tables with names and fields
    # tables1.columns = ["Name", "Field", "Date"]
    #
    # url = r'http://www.massey.ac.nz/~iabond/moa/alert%s/alert.php/' % (year) # Table with names and coordinates
    # tables2 = pd.read_html(url)  # Returns list of all tables on page
    # nan_value = float("NaN")  # Nan value to convert from empty string
    # tables2[0].replace("", nan_value, inplace=True)
    # alert_table = tables2[0].dropna(subset=["RA (J2000.0)", "Dec (J2000.0)"])
    #
    # catalog = SkyCoord(ra=alert_table["RA (J2000.0)"],
    #                    dec=alert_table["Dec (J2000.0)"], unit=(u.hourangle, u.deg))
    #
    # idxc, idxcatalog, d2d, d3d = catalog.search_around_sky(c, rad * u.degree)
    #
    # for i in range(len(idxc)):
    #     name = alert_table["ID"].values[idxcatalog[i]]
    #     field = tables1.loc[tables1["Name"] == name]["Field"]
    #     xmatch_result.append(
    #         (names[idxc[i]], "MOA-%s"%name, field))

    return xmatch_result

def asassn_xmatch(names, ra, dec):
    """
    Returns a list of Gaia alert names and corresponding ASASSN alerts.

    Args:
        - names string - name of Gaia alert
        - ra, dec   float - coordinates on the sky in equatorial frame, degrees
    Returns:
        - xmatch_result list - Gaia alert name (first element) and corresponding ASASSN alert name (second element)
    """
    rad = 4.0 / 60. / 60.  # 0.5 arcsec search radius
    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    xmatch_result = []

    url = r'http://www.astronomy.ohio-state.edu/asassn/transients.html'
    tables = pd.read_html(url)  # Returns list of all tables on page
    nan_value = float("NaN")  # Nan value to convert from empty string
    tables[0].replace("", nan_value, inplace=True)
    alert_table = tables[0].dropna(subset=["RA", "Dec"])

    catalog = SkyCoord(ra=alert_table["RA"].values[:],
                       dec=alert_table["Dec"].values[:], unit=(u.hourangle, u.deg))

    idxc, idxcatalog, d2d, d3d = catalog.search_around_sky(c, rad * u.deg)

    for i in range(len(idxc)):
        asasn_name = tables[0]["ASAS-SN"].values[idxcatalog[i]]
        if(asasn_name != "---"):
            xmatch_result.append((names[idxc[i]], asasn_name))
        else:
            xmatch_result.append((names[idxc[i]], "in ASAS-SN"))

    return xmatch_result

def exclude_KMTNet_fields(names, ra, dec):
    KMTNet_fields = np.array([[264.00, -37.40],
                        [270.50, -37.40],
                        [270.50, -33.75],
                        [272.50, -33.75],
                        [272.50, -32.00],
                        [275.50, -32.00],
                        [275.50, -25.30],
                        [275.60, -25.30],
                        [275.60, -21.90],
                        [272.00, -21.90],
                        [272.00, -23.00],
                        [270.40, -23.00],
                        [270.40, -20.50],
                        [264.50, -20.50],
                        [264.50, -22.70],
                        [262.00, -22.70],
                        [262.00, -26.25],
                        [260.50, -26.25],
                        [260.50, -31.40],
                        [262.00, -31.40],
                        [262.00, -36.00],
                        [264.00, -36.00]])
    remove_idx = []

    exclusion_zone = Polygon(zip(KMTNet_fields[:,0], KMTNet_fields[:,1]))
    for i in range(len(names)):
        point = Point(ra[i], dec[i])
        if(exclusion_zone.contains(point)):
            print("%s: Within KMTNet exclusion zone."%(names[i]))
            remove_idx.append(i)

    return np.array(remove_idx)
