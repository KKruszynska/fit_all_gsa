import sys

from datetime import datetime, timedelta

import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.signal import savgol_filter

from pyLIMA import event
from pyLIMA import telescopes
from pyLIMA.fits import TRF_fit
from pyLIMA.models import PSPL_model

# import module_findingcharts, module_gaia_ulens
from module_gaia_xmatch import ogle_xmatch, kmtn_xmatch, moa_xmatch, asassn_xmatch, exclude_KMTNet_fields
from module_ulens_plots import plot_lightcurve, plot_data, plot_sdss, plot_xmatchps1jpeg
import module_ulens_data as mud

# Defalut settings: events from last two weeks
time_max = datetime.today()
time_min = time_max - timedelta(days=14)
base_file_output_name = "new_candidates"
hcz_zone = True

if (len(sys.argv) not in [4, 5] and len(sys.argv) != 1):
    print("Wrong number of arguments. Provide start date, end date and output file name base.")
    print("Run program as follows:")
    print("python fit_all_gsa_py3.py yyyy-mm-dd yyyy-mm-dd output_file_name_base")
    print("Or, to turn off high-cadence zone exlusion:")
    print("python fit_all_gsa_py3.py yyyy-mm-dd yyyy-mm-dd output_file_name_base hcz_zone=False")
    quit()
if(len(sys.argv) == 4):
    time_min = sys.argv[1] # has to be yyyy-mm-dd
    time_max = sys.argv[2] # has to be yyyy-mm-dd
    base_file_output_name = sys.argv[3]
if(len(sys.argv) == 5):
    time_min = sys.argv[1] # has to be yyyy-mm-dd
    time_max = sys.argv[2] # has to be yyyy-mm-dd
    base_file_output_name = sys.argv[3]
    if (sys.argv[4].split("=")[1] == 'False'):
        hcz_zone = False
    else:
        print("High cadence zone not turned off!")
        print("To turn off high-cadence zone exlusion:")
        print("python fit_all_gsa_py3.py yyyy-mm-dd yyyy-mm-dd output_file_name_base hcz_zone=False")
        quit()

# find events in the required time range
names, ra, dec = mud.select_gsa_events(time_min, time_max)
print("Candidate names:")
print(names)

#exclude KMTNet events -- for OMEGA
idx_in_kmtn_zone = exclude_KMTNet_fields(names, ra, dec)

#cross-match with other surveys (OGLE, MOA, KMTNet, ASASSN)
ogle_names = np.asarray(ogle_xmatch(names, ra, dec))
kmtn_names = np.asarray(kmtn_xmatch(names, ra, dec))
moa_names = np.asarray(moa_xmatch(names, ra, dec))
asassn_names = np.asarray(asassn_xmatch(names, ra, dec))

# #SETUP
searchradnei = 0.7 / 60 / 60  # neighbours #was 0.1
searchradsdsssatur = 30. / 60 / 60.
searchradgal = 13. / 60 / 60.  # deg

# opening output file
num_file = 1
count = 0
output_file_name = "%s_part_%d.html" % (base_file_output_name, num_file)
print("Num file: ", num_file)
output = open(output_file_name, "w")

comment_hcz = ""
for i in range(len(names)):
    if(hcz_zone):
        if (i in idx_in_kmtn_zone):
            print("%s: Within KMTNet exclusion zone"%(names[i]))
            continue
    else:
        if (i in idx_in_kmtn_zone):
            comment_hcz = '<font color="red"><b>In HCZ</b></font>'
        else:
            comment_hcz = '<font color="black">Outside HCZ</font>'


    if(i % 100 == 0):
        output_file_name = "%s_part_%d.html" % (base_file_output_name, num_file)
        print("Num file: ", num_file)
        output = open(output_file_name, "w")
        num_file += 1
        output.write(
            "Content-Type: text/html;charset=utf-8 \n \
            <HTML><HEAD><TITLE>Microlensing Candidates part %d</TITLE></HEAD> \n<BODY>\n"
            % num_file)
        # currentdate = (time.strftime("%d/%m/%Y %H:%M:%S %Z"))
        currentdate = datetime.utcnow()
        output.write("%s UTC \n<br>\n" % currentdate)
        output.write('<font color="#4928e0"><b>abs(b)<7.5 deg </b></font>, '
                     '<font color="#b81f69"><b>7.5 deg < abs(b) < 15 deg</b></font>, '
                     '<font color="#d65302"><b>abs(b)>15 deg</b></font>')
        output.write(
            '<table border="1" style="width:1200px">\n <tr><th>count</th>\n <th>i</th>\n <th>pub name</th>\n <th>light curve</th>\n <th>lc with follow-up</th>\n' \
            '<th>Last ~200 days</th>\n <th>ulens fit</th>\n <th>Images</th>\n <th>ra[deg]</th>\n <th>dec[deg]</th>\n <th>l [deg]</th>\n <th>b [deg]</th>\n')

    alpha = ra[i]
    delta = dec[i]

    ## computing equarial and galactic coordinates
    c = SkyCoord(ra=alpha * u.degree, dec=delta * u.degree)
    gall = c.galactic.l.degree
    galb = c.galactic.b.degree

    ## computing equarial and galactic coordinates
    publishedas = names[i]

    # Gaia
    gaia_times, gaia_mags = mud.get_lightcurve_gaia(publishedas)
    if (len(gaia_mags) < 10):
        continue
    gaia_errs = np.array(mud.get_gaia_errors(gaia_mags))

    if (abs(galb) < 7.5):
        output.write('<tr>\n <td><font color="#785EF0"><b>%d</b></font></td>\n' % (count))
    elif (abs(galb) < 15.):
        output.write('<tr>\n <td><font color="#DC267F"><b>%d</b></font></td>\n' % (count))
    else:
        output.write('<tr>\n <td><font color="#FE6100"><b>%d</b></font></td>\n' % (count))

    output.write('<td><b>%d</b></td>\n' % (i))
    output.write('<td><a href="https://gsaweb.ast.cam.ac.uk/alerts/alert/%s">%s</a>\n' % (publishedas, publishedas))
    output.write('<br>%s\n' % (comment_hcz))

    ogle_times, ogle_mags, ogle_errs = 0,0,0
    if len(ogle_names) > 0 and publishedas in ogle_names[:,0]:
        indexes = np.where(ogle_names[:, 0] == publishedas)
        for idx in indexes:
            survey_name = ogle_names[idx, 1][0]
            ogle_times, ogle_mags, ogle_errs = mud.get_lightcurve_OGLE_EWS(survey_name)
            text = survey_name.split('-')
            year = text[1]
            num = text[3]
            output.write("<br><a href='http://ogle.astrouw.edu.pl/ogle4/ews/%s/blg-%s.html'>%s</a>"%(year, num, survey_name))

    moa_times, moa_mags, moa_errs = 0,0,0
    if len(moa_names) > 0 and publishedas in moa_names[:, 0]:
        indexes = np.where(moa_names[:, 0] == publishedas)
        for idx in indexes:
            survey_name = moa_names[idx, 1][0]
            print(survey_name)
            text = survey_name.split('-')
            year = text[1]
            field = moa_names[idx, 2][0]
            # moa_times, moa_mags, moa_errs = mud.get_lightcurve_MOA(survey_name, field)
            output.write("<br><a href='http://www.massey.ac.nz/~iabond/moa/alert%s/display.php?id=%s'>%s</a>\n"%(year, field, survey_name))

    if len(asassn_names) > 0 and publishedas in asassn_names[:, 0]:
        indexes = np.where(asassn_names[:, 0] == publishedas)
        for idx in indexes:
            survey_name = asassn_names[idx, 1]
            output.write("<br>%s" %(survey_name))

    output.write('</td>')

    # Get data from surveys
    # ZTF
    ztf_dr_data, ztf_band = 0, 0
    ztf_dr_data, ztf_band = mud.get_lightcurve_ZTF_DR(alpha, delta)
    # fol, ztf_al_data, ztf_al_bands = mud.get_lightcurve_ZTF_alert(alpha, delta)

    # Microlensing model
    # Setting up a pyLIMA events
    # Gaia only event
    # gaia_event = event.Event()
    # gaia_event.name = publishedas
    # gaia_event.ra, gaia_event.dec = alpha, delta
    # print("Gaia only event created.")

    # Gaia and follow-up (first we check if data is available)
    gaia_fup_event = event.Event()
    gaia_fup_event.name = publishedas
    gaia_fup_event.ra, gaia_fup_event.dec = alpha, delta
    print("Gaia + fup event created.")

    # Time to load data and add telescopes...
    datasets, telescope_labels = [], []
    n_telescopes = 0
    # Gaia
    gaia_data = np.vstack((gaia_times, gaia_mags, gaia_errs))
    t_last = gaia_times[-1]
    datasets.append(gaia_data)
    telescope_labels.append("Gaia")
    telescope_gaia = telescopes.Telescope(name='Gaia',
                                          camera_filter='G',
                                          light_curve=(gaia_times, gaia_mags, gaia_errs),
                                          light_curve_names=['time', 'mag', 'err_mag'],
                                          light_curve_units=['JD', 'mag', 'err_mag'],
                                          location='Space', spacecraft_name='Gaia')
    # gaia_event.telescopes.append(telescope_gaia)
    gaia_fup_event.telescopes.append(telescope_gaia)
    n_telescopes += 1
    print("%s: %s: Gaia data added.\n" % (datetime.utcnow(), publishedas))

    # MOA
    # if (type(moa_times) is not int):
    #     moa_data = np.vstack((moa_times, moa_mags, moa_errs))
    #     datasets.append(moa_data)
    #     telescope_labels.append("MOA")
    #     if(t_last < moa_times[-1]):
    #         t_last = moa_times[-1]
    #
    #     telescope_moa = telescopes.Telescope(name='MOA',
    #                                          light_curve=(moa_times, moa_mags, moa_errs),
    #                                          light_curve_names=['time', 'flux', 'err_flux'],
    #                                          light_curve_units=['JD', 'flux', 'err_flux'],
    #                                          location='Earth')
    #     gaia_fup_event.telescopes.append(telescope_moa)
    #     n_telescopes += 1
    #     print("%s : %s: MOA data added.\n" % (datetime.utcnow(), publishedas))

    # OGLE EWS
    if (type(ogle_times) is not int):
        ogle_data = np.vstack((ogle_times, ogle_mags, ogle_errs))
        datasets.append(ogle_data)
        telescope_labels.append("OGLE")
        if (t_last < ogle_times[-1]):
            t_last = ogle_times[-1]
        telescope_ogle = telescopes.Telescope(name='OGLE',
                                             camera_filter='I',
                                             light_curve=(ogle_times, ogle_mags, ogle_errs),
                                             light_curve_names=['time', 'mag', 'err_mag'],
                                             light_curve_units=['JD', 'mag', 'err_mag'],
                                             location='Earth')
        gaia_fup_event.telescopes.append(telescope_ogle)
        n_telescopes += 1
        print("%s : %s: OGLE EWS data added.\n" % (datetime.utcnow(), publishedas))

    # ZTF DR
    if (type(ztf_dr_data) is not int):
        b = 0
        for zdata in ztf_dr_data:
            ztime, zmags, zerrs = zdata[:,0], zdata[:,1], zdata[:,2]
            if(type(ztime) is not float):
                data = np.vstack((ztime, zmags, zerrs))
                datasets.append(data)
                telescope_labels.append("ZTF_"+ztf_band[b])
                if (t_last < ztime[-1]):
                    t_last = ztime[-1]
                telescope_ztf_dr = telescopes.Telescope(name='ZTF_'+ztf_band[b],
                                                     camera_filter=ztf_band[b],
                                                     light_curve=(ztime, zmags, zerrs),
                                                     light_curve_names=['time', 'mag', 'err_mag'],
                                                     light_curve_units=['JD', 'mag', 'err_mag'],
                                                     location='Earth')
                gaia_fup_event.telescopes.append(telescope_ztf_dr)
                n_telescopes += 1
            b += 1
        print("%s : %s: ZTF Data Release data added.\n" % (datetime.utcnow(), publishedas))

    # ZTF Alerts
    # if (fol > 0):
    #     b = 0
    #     for zdata in ztf_al_data:
    #         if (t_last < zdata[0,-1]):
    #             t_last = zdata[0,-1]
    #         telescope_ztf_al = telescopes.Telescope(name='ZTF',
    #                                                 camera_filter=ztf_al_bands[b],
    #                                                 light_curve=zdata,
    #                                                 light_curve_names=['time', 'mag', 'err_mag'],
    #                                                 light_curve_units=['JD', 'mag', 'err_mag'],
    #                                                 location='Earth')
    #         gaia_fup_event.telescopes.append(telescope_ztf_al)
    #     print("%s : %s: ZTF Alerts data added.\n" % (datetime.utcnow(), publishedas))

    # Finally, lets check if the event looks fine...
    # gaia_event.find_survey('Gaia')
    # gaia_event.check_event()
    gaia_fup_event.find_survey('Gaia')
    gaia_fup_event.check_event()

    try:
        # Initial guess
        data_smooth = savgol_filter(gaia_mags, 14, 4)
        t0guess = gaia_mags[np.argmin(data_smooth)]

        # Fit PSPL without blending to Gaia only data
        # pspl_gaia = PSPL_model.PSPLmodel(gaia_event, parallax=['Full', t0guess], blend_flux_parameter='noblend')
        # gaia_fit = TRF_fit.TRFfit(pspl_gaia)
        # gaia_fit.fit()

        # Fit PSPL without blending to Gaia only data

        pspl_fup = PSPL_model.PSPLmodel(gaia_fup_event, parallax=['Full', t0guess], blend_flux_parameter='noblend')
        fup_fit = TRF_fit.TRFfit(pspl_fup)
        fup_fit.fit()
        best_params = fup_fit.fit_results['best_model']
        chi2 = fup_fit.fit_results['chi2']
    except:
        print("exeption happened: fitting")
        best_params = np.zeros(6)
        chi2 = 0.

    # Plots
    # times for plot
    print("Number of telescopes:", n_telescopes)
    print("Labels:", telescope_labels)
    tmin, tmax = gaia_times[0]-50.-2450000., t_last+70.-2450000.
    if(n_telescopes == 1):
        gaia_uri = plot_data(publishedas, datasets[0], 1, telescope_labels, tmin, tmax)
        fup_uri = plot_data(publishedas, datasets[0], n_telescopes, telescope_labels, tmin, tmax)
        zoom_uri = plot_data(publishedas, datasets[0], n_telescopes, telescope_labels, tmax-270, tmax)
    else:
        gaia_uri = plot_data(publishedas, datasets[0], 1, telescope_labels, tmin, tmax)
        fup_uri = plot_data(publishedas, datasets, n_telescopes, telescope_labels, tmin, tmax)
        zoom_uri = plot_data(publishedas, datasets, n_telescopes, telescope_labels, tmax-270, tmax)

    model_uri = ""
    try:
        model_uri = plot_lightcurve(publishedas, fup_fit, pspl_fup, tmin + 2450000. - 200., tmax + 2450000. + 30.)
    except:
        print("exeption happened")

    output.write('<td>')
    output.write('<IMG src = "%s" width="400" height="300"/>' % gaia_uri)
    output.write('</td>')

    output.write('<td>')
    output.write('<IMG src = "%s" width="400" height="300"/>' % (fup_uri))
    output.write('</td>\n')

    lineout = ('<td><IMG src = "%s" width="400" height="300">') % (zoom_uri)
    output.write(lineout)
    output.write('</td>\n')

    if(len(model_uri)>0):
        lineout = ('<td><IMG src = "%s" width="400" height="300"><br>fit: t0=%f u0=%f tE=%f piEN=%f \
        piEE= %f Fs=%f Fb=%f chi2=%f')%(
            model_uri, best_params[0], best_params[1], best_params[2], best_params[3],
            best_params[4], best_params[5], 0.0, chi2)
    else:
        lineout = '<td>Failed to find valid microlensing model.'
    output.write(lineout)
    output.write('</td>\n')

    # SDSS
    output.write(
        '<td><a href="http://skyserver.sdss3.org/dr13/en/tools/chart/navi.aspx?ra=%s&dec=%s">' % (alpha, delta))
    output.write('<IMG SRC = "%s"/>' % plot_sdss(alpha, delta, 150, 150))
    output.write('</a><br>')
    output.write('<a href="http://archive.stsci.edu/panstarrs/search.php?ra=%s&dec=%s&action=Search&radius=0.5">' % (
    alpha, delta))
    # output.write('<img src = "%s"/></a></td>\n' % plot_xmatchps1jpeg(alpha, delta))

    ### additional info on the alert later
    output.write('<td>%f</td>\n' % (alpha))
    output.write('<td>%f</td>\n' % (delta))
    output.write("<td>%f</td>\n" % (gall))
    output.write("<td>%f</td>\n" % (galb))
    output.write("</tr>\n")

    if (count % 100 == 99):
        output.write('</table>\n')
        output.write('FINISHED<br><br>')
        output.write("</BODY></HTML>")
        output.close()
    count += 1

output.write('</table>\n')
output.write('FINISHED<br><br>')

output.write("</BODY></HTML>")
output.close()
