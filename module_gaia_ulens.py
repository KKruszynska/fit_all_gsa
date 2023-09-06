# module for Gaia microlensing fits and plots

import time, calendar #for displaying date/time
from datetime import datetime

import psycopg2

import math,sys

import string

import numpy as np
import scipy.stats
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import io
import urllib, base64
import matplotlib.gridspec as gridspec
import json
import mechanize
import requests
import urllib 

# import alert_single_img
import ulensparallax
from etienne_extract_one_alert_radec import getztfmars

mjdoffset = 2456000.
jdoffset2 = 2400000.
HPLEVEL=5
nSide = 2**HPLEVEL
GaiaNSide = 4096
OBT_MASK = 0b00000111111111111111111111111111111111111111111
OBT_FACTOR = 4096 * 50

lastsun = np.genfromtxt("last_monday.dat")

### ulens_fiexbl(t, t0, te, u0, I0)
# Function by LW, from ulensparallax.py module.
# Copied here, because ulensparallax.py requires parsubs.so file
# Input parameters: t  : 1dim table of floats representing time
#                   t0 : time of peak of brightness
#                   te : Einstein time, float
#                   u0 : impact parameter, float
#                   I0 : baseline magnitude, float
# Output:           returns 1dim table of floats of magnitudes of an ulens with blending fixed at fs=1.0
# Use:              Classical lens, no effects, fixed blending
def ulens_fixedbl(t, t0, te, u0, I0):
    #print '<br>', type(t), type(t0), type(te)
    #print '<br>', t-t0, te
    tau=(t-t0)/te
    x=tau
    y=u0

    u=np.sqrt(x**2+y**2)
    ampl= (u**2 + 2)/(u*np.sqrt(u**2+4))
    F = ampl
    I = I0 - 2.5*np.log10(F)
    return I

# Error for multiple datasets for PSPL model (fixedbl)
def error_pspl(v, datasets, ndatasets):
    fp = lambda v, I0, x: ulens_fixedbl(x, v[0],v[1],v[2], I0)
    e = lambda v, I0, x, y, bar: ((fp(v,I0,x)-y)/bar)
    all_errors = []
    for i in range(ndatasets):
        I0 = v[i+2]
        times = np.array(datasets[i][0][:])
        mag = np.array(datasets[i][1][:])
        err = np.array(datasets[i][2][:])
        error = e(v, I0, times, mag, err)
        all_errors = np.concatenate([all_errors, error])
    #print all_errors
    return all_errors

# Error for multiple datasets for parallax model (fixedbl)
def error_par(v, datasets, ndatasets, alpha, delta, t0par):
    fp = lambda v, I0, x: ulensparallax.ulensparallax_fixedbl(x, alpha, delta, t0par, v[0], v[1], v[2], v[3], v[4], I0)
    e = lambda v, I0, x, y, bar: ((fp(v,I0,x)-y)/bar)
    all_errors = []
    for i in range(ndatasets):
        I0 = v[i+4]
        time = np.array(datasets[i][0][:])
        mag = np.array(datasets[i][1][:])
        err = np.array(datasets[i][2][:])
        error = e(v, I0, time, mag, err)
        all_errors = np.concatenate([all_errors, error])
    return all_errors

### fit_ulensfixedbl(epoch, avmag, err)
# Input parameters: epoch : 1dim table with time of observation of each point
#                   avmag : 1dim table with average measurement of brightness in mag from all AFs
# Output:           out   : 1dim vector with best fitting values t0-2450000,te,u0,I0 and chi2 and chi2/dof (in this order)
# Use:              Fitting ulensing model with fixed blending at fs=1. Returns a vector with best
#                   fitting parameters (t0-2450000,te,u0,I0) and chi2 and chi2/dof.
#                   Error of each brightness measurement is fixed. Please change if needed.
def fit_ulensfixedbl_single(epoch, avmag, err):
    t0=epoch[np.argmin(avmag)]
    te=100.
    u0=0.05
    I0=np.amax(avmag)
    x=epoch
    y=avmag
    
    ulensparam=[t0, te, u0, I0]
    fp = lambda v, x: ulens_fixedbl(x, v[0],v[1],v[2],v[3])
    e = lambda v, x, y, err: ((fp(v,x)-y)/err)
    v, success = leastsq(e, ulensparam, args=(x,y,err), maxfev=1000000)
    #    print v, success
    chi2 = sum(e(v,x,y,err)**2)
    chi2dof=sum(e(v,x,y,err)**2)/(len(x)-len(v))
    out = []
    for t in v:
        out.append(t)
    return out, chi2dof

#fit PSPL model with fixd blending to multiple datasets    
def fit_ulensfixedbl_multi(data, ndata):
    t0=data[0][0][np.argmin(data[0][1][:])]
    te=100.
    u0=0.05
    ulensparam=[t0, te, u0]
    for i in range(ndata):
        ulensparam.append(np.amax(data[i][1][:]))

    v, success = leastsq(error_pspl, ulensparam, args=(data, ndata), maxfev=1000000)
    #    print v, success
    chi2 = sum(error_pspl(v, data, ndata)**2)
    lentime = 0
    for i in range(ndata):
        lentime = lentime + len(data[i][0][:])
    chi2dof=chi2/(lentime-len(v))
    out = []
    for t in v:
        out.append(t)
    return out, chi2dof

def fit_ulensparallax(epoch, avmag, err, alpha, delta, v1):
    t0 = v1[0]
    t0par = t0
    te = v1[1]
    u0 = v1[2]
    I0 = v1[3]
    fs = 1.0
    piEE = 0.0
    piEN = 0.0
    x = epoch
    y = avmag
    
    #ulensparallax(t, alpha, delta, t0par, t0, te, u0, piEN, piEE, I0, fs)
    ulensparam = [t0, te, u0, piEN, piEE, I0, fs]
    fp = lambda v, x: ulensparallax.ulensparallax(x, alpha, delta, t0par, v[0],v[1],v[2],v[3], v[4],v[5],v[6])
    e = lambda v, x, y, err: ((fp(v,x)-y)/err)
    v, success = leastsq(e, ulensparam, args=(x,y,err), maxfev=1000000)
    #    print v, success
    chi2 = sum(e(v,x,y,err)**2)
    chi2dof=sum(e(v,x,y,err)**2)/(len(x)-len(v))
    out = []
    for t in v:
        out.append(t)
    return out, chi2dof
    

def fit_ulensparallax_fixedbl_single(epoch, avmag, err, alpha, delta,v1):
    t0 = v1[0]
    t0par = t0
    te = v1[1]
    u0 = v1[2]
    I0 = v1[3]
    piEE = 0.0
    piEN = 0.0
    x = epoch
    y = avmag
    
    #ulensparallax(t, alpha, delta, t0par, t0, te, u0, piEN, piEE, I0, fs)
    ulensparam = [t0, te, u0, piEN, piEE, I0]
    fp = lambda v, x: ulensparallax.ulensparallax_fixedbl(x, alpha, delta, t0par, v[0],v[1],v[2],v[3], v[4],v[5])
    e = lambda v, x, y, err: ((fp(v,x)-y)/err)
    v, success = leastsq(e, ulensparam, args=(x,y,err), maxfev=1000000)
#    print v, success
    chi2 = sum(e(v,x,y,err)**2)
    chi2dof=sum(e(v,x,y,err)**2)/(len(x)-len(v))
    out = []
    for t in v:
                out.append(t)
    return out, chi2dof
    
# Fit parallax model with fixed blending to a set of observational data
def fit_ulensparallax_fixedbl_multi(data, ndata, alpha, delta, v1):
    t0 = v1[0]
    t0par = t0
    te = v1[1]
    u0 = v1[2]
    piEE = 0.0
    piEN = 0.0
    
    #ulensparallax(t, alpha, delta, t0par, t0, te, u0, piEN, piEE, I0, fs)
    ulensparam = [t0, te, u0, piEN, piEE]
    for i in range(ndata):
        ulensparam.append(v1[i+2])
    
    v, success = leastsq(error_par, ulensparam, args=(data, ndata, alpha, delta, t0par), maxfev=1000000)
    #    print v, success
    chi2 = sum(error_par(v, data, ndata, alpha, delta, t0par)**2)
    lentime = 0
    for i in range(ndata):
        lentime = lentime + len(data[i][0][:])
    chi2dof=chi2/(lentime-len(v))
    out = []
    for t in v:
        out.append(t)
    return out, chi2dof

### gen_modellightcurve(epoch, avmag, v) < -- based on alerts-ulensing-mass.py
# Input parameters: epoch : 1dim table with time
#                   avmag : 1dim table with average measurement of brightness in mag from all AFs
#                   v     : 1dim vector with parameters of PSPL model t0-2450000,te,u0,I0 and chi2 and chi2/dof (in this order)
# Output:           uri   : buffered image, which can be inserted into a html file
# Use:              This function generates a lightcurve of a PSPL model best fitting to data, which is represented by (epoch, avmag)
#                   Error of each brightness measurement is fixed. Please change if needed.
def gen_modellightcurve(epoch, avmag, v, GaiaZeroPoint):
    x = epoch
    y = avmag
    err = np.full(len(epoch), 0.02) # fixed value of error, please change if needed
    
    fp = lambda v, x: ulens_fixedbl(x, v[0],v[1],v[2],v[3])
    
    xlc=np.linspace(x[0],x[-1],1000)
    ylc=fp(v,xlc)
    
    fig=plt.figure(figsize=(8,4)) # figsize is in inches
    ax = plt.subplot(111)
    plt.errorbar(x,y,yerr=err,fmt='ro') # lightcurve for a given source
    plt.errorbar(xlc,ylc,fmt='g-') # PSPL model with fixed blending at fa=1.0
    plt.title("%s"%name)
    locs,labels = plt.xticks()
    plt.xticks(locs, map(lambda x: "%g" % x, locs))
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%g" % x, locs))
    plt.xlabel("JD-2450000")
    
    plt.gca().invert_yaxis()
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue())
    html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
    img.close()
    plt.close(fig)
    
    return html_code

def gen_modellightcurve_par(name,epoch, avmag, v, v2, alpha, delta, GaiaZeroPoint):
    x = epoch
    y = avmag
    t0par = v[0]
    err = np.full(len(epoch), 0.02) # fixed value of error, please change if needed

    fp = lambda v, x: ulens_fixedbl(x, v[0],v[1],v[2],v[3])
    fp2 = lambda v2, x: ulensparallax.ulensparallax(x, alpha, delta, t0par, v2[0],v2[1],v2[2],v2[3],v2[4],v2[5],v2[6])

    xlc=np.linspace(x[0],x[-1],1000)
    ylc=fp(v,xlc)
    ylc2=fp2(v2,xlc)
    gs = gridspec.GridSpec(3,1)
    fig=plt.figure(figsize=(8,4)) # figsize is in inches
    ax = plt.subplot(gs[:2, :])
    plt.gca().invert_yaxis()
    plt.title("%s"%name)
    plt.errorbar(x,y,yerr=err,fmt='ro') # lightcurve for a given source
    plt.errorbar(xlc,ylc,fmt='g-') # PSPL model with fixed blending at fa=1.0
    plt.errorbar(xlc,ylc2,fmt='b-') #parallax model with unfixed blending

    #get followup
    datajson, followup = get_followup(name)
    ampl = np.amax(mags) - np.amin(mags)
    if (followup != 0):
        mjd0=np.array(datajson['mjd'])
        mag0=np.array(datajson['mag'])
        magerr0=np.array(datajson['magerr'])
        filter0=np.array(datajson['filter'])
        caliberr0=np.array(datajson['caliberr'])
        obs0 = np.array(datajson['observatory'])

        for band in filter_list:
            indexes = np.where(filter0 == band)
            ftime = mjd0[indexes] + jdoffset2 - 2450000.
            fmags = mag0[indexes]
            ferr = magerr0[indexes]
            fv, fchi2dof = fit_ulensfixedbl(ftime, fmags, ferr)
            fflux = 10**((GaiaZeroPoint - fmags)/2.5)
            model_mag = fp(v,ftime)
            model_flux = 10**((GaiaZeroPoint - model_mag)/2.5)
            fres = 2.5*np.log10(model_flux/fflux)
            norm_mags = model_mag + res
            plt.errorbar(ftime,norm_mags,yerr=ferr,fmt='.')

    ax2 = plt.subplot(gs[2:, :])
    plt.errorbar(xlc,ylc2-ylc,fmt='b-')
    plt.errorbar(x,y-fp(v,x),yerr=err,fmt='ro')
    plt.axhline(y=0)



    locs,labels = plt.xticks()
    plt.xticks(locs, map(lambda x: "%g" % x, locs))
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%g" % x, locs))
    plt.xlabel("JD-2450000")

    plt.gca().invert_yaxis()
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue())
    html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
    img.close()
    plt.close(fig)

    return html_code

def gen_modellightcurve_par_fixedbl_single(name, datasets, v, v2, alpha, delta, GaiaZeroPoint, tmin, tmax):
    x = datasets[0][0][:]
    y = datasets[0][1][:]
    t0par = v[0]
    err = datasets[0][2][:]

    fp = lambda v, x: ulens_fixedbl(x, v[0],v[1],v[2],v[3])
    fp2 = lambda v2, x: ulensparallax.ulensparallax_fixedbl(x, alpha, delta, t0par, v2[0],v2[1],v2[2],v2[3],v2[4],v2[5])

    gs = gridspec.GridSpec(3,1)
    fig=plt.figure(figsize=(8,4)) # figsize is in inches

    xlc=np.linspace(x[0]-500.,x[-1]+500.,10000)
    ylc=fp(v,xlc)
    ylc2=fp2(v2,xlc)
    ax = plt.subplot(gs[:2, :])
    plt.title("%s"%name)
    if(len(datasets[:]) > 1):
        for i in range(len(datasets[:])):
            ftime = np.array([datasets[i][0][:]])
            fmags = np.array(datasets[i][1][:])
            ferr = np.array(datasets[i][2][:])
            if (len(ftime)<4):
                    continue
            else:
                fv, fchi2dof = fit_ulensfixedbl_single(ftime, fmags, ferr)
                fflux = 10.**((GaiaZeroPoint-fmags)/-2.5)
                fflux_model = 10.**((GaiaZeroPoint-fp(fv,ftime))/-2.5)
                norm_mags = fp(v, ftime) + 2.5*np.log10(fflux_model/fflux)
                #print name, band, len(ftime), fmags[0], '</br>'
                plt.errorbar(ftime,norm_mags,yerr=ferr,fmt='.')

    #plt.gca().invert_yaxis()
    plt.errorbar(x,y,yerr=err,fmt='ro', zorder=12) # lightcurve for a given source
    plt.errorbar(xlc,ylc,fmt='g-', zorder=9) # PSPL model with fixed blending at fs=1.0
    plt.errorbar(xlc,ylc2,fmt='b-', zorder=10) #parallax model with fixed blending
    plt.axvline(x=lastsun,ls='--', color='black') # last Monday
    plt.xlim(tmin, tmax)
    plt.ylim(v[3]+0.6, min(np.array(datasets[0][1][:]))-0.6)

    ax2 = plt.subplot(gs[2:, :])
    plt.errorbar(xlc,ylc2-ylc,fmt='b-', zorder=10)
    plt.errorbar(x,y-fp(v,x),yerr=err,fmt='ro', zorder=12)
    if(len(datasets[:]) > 1):
        for i in range(len(datasets[:])):
            ftime = np.array([datasets[i][0][:]])
            fmags = np.array(datasets[i][1][:])
            ferr = np.array(datasets[i][2][:])
            if (len(ftime)<4):
                continue
            else:
                fv, fchi2dof = fit_ulensfixedbl_single(ftime, fmags, ferr)
                plt.errorbar(ftime,fmags - fp(fv,ftime),yerr=ferr,fmt='.')

    plt.axhline(y=0, color='g', zorder=9)
    plt.axvline(x=lastsun,ls='--', color='black') # last Monday
    plt.xlim(tmin, tmax)

    #locs,labels = plt.xticks()
    #plt.xticks(locs, map(lambda x: "%g" % x, locs))
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%g" % x, locs))
    plt.xlabel("JD-2450000")
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue())
    html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
    img.close()
    plt.close(fig)

    return html_code

def gen_modellightcurve_par_fixedbl_multi(name, data, ndata, v, v2, alpha, delta, GaiaZeroPoint):
    x = data[0][0][:]
    y = data[0][1][:]
    err = data[0][2][:]
    t0par = v[0]

    fp = lambda v, x: ulens_fixedbl(x, v[0],v[1],v[2],v[3])
    fp2 = lambda v2, x: ulensparallax.ulensparallax_fixedbl(x, alpha, delta, t0par, v2[0],v2[1],v2[2],v2[3],v2[4],v2[5])

    xlc=np.linspace(x[0]-500.,x[-1]+500.,10000)
    ylc=fp(v,xlc)
    ylc2=fp2(v2,xlc)

    gs = gridspec.GridSpec(3,1)
    fig=plt.figure(figsize=(8,4)) # figsize is in inches
    ax2 = plt.subplot(gs[:2, :])
    plt.gca().invert_yaxis()
    plt.title("%s"%name)
    plt.errorbar(x,y,yerr=err,fmt='ro', zorder=12) # lightcurve for a given source
    plt.errorbar(xlc,ylc,fmt='g-', zorder=9) # PSPL model with fixed blending at fs=1.0
    plt.errorbar(xlc,ylc2,fmt='b-', zorder=10) #parallax model with fixed blending
    v0 = [v[0], v[1], v[2], v[3]]

    for i in range(2, ndata):
        ftime = np.asarray(data[i][0][:])
        fmags = np.asarray(data[i][1][:])
        ferr = np.asarray(data[i][2][:])
        fv = [v[0], v[1], v[2], v[3+i]]
        print('<br>', fv[3])
        fflux = 10.**((GaiaZeroPoint-fmags)/-2.5)
        fflux_model = 10.**((GaiaZeroPoint-fp(fv,ftime))/-2.5)
        norm_mags = fp(v0, ftime) + 2.5*np.log10(fflux_model/fflux)
        #print name, band, len(ftime), fmags[0], '</br>'
        plt.errorbar(ftime,norm_mags,yerr=ferr,fmt='.')
                                
    ax2 = plt.subplot(gs[2:, :])
    plt.errorbar(xlc,ylc2-ylc,fmt='b-', zorder=10)
    plt.errorbar(x,y-fp(v0,x),yerr=err,fmt='ro', zorder=12)

    for i in range(2, ndata):
        # plot plot plot
        ftime = data[i][0][:]
        fmags = data[i][1][:]
        ferr = data[i][2][:]
        fv = [v[0], v[1], v[2], v[2+i]]
        plt.errorbar(ftime,fmags - fp(fv,ftime),yerr=ferr,fmt='.')
    plt.axhline(y=0, color='g', zorder=9)


    locs,labels = plt.xticks()
    plt.xticks(locs, map(lambda x: "%g" % x, locs))
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%g" % x, locs))
    plt.xlabel("JD-2450000")

    plt.gca().invert_yaxis()


    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue())
    html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
    img.close()
    plt.close(fig)

    return html_code

def gen_followup_single(name,times, mags, err, tmin, tmax):	
    fig=plt.figure(figsize=(8,4)) # figsize is in inches
    plt.gca().invert_yaxis()
    plt.errorbar(times,mags,yerr=err,fmt='ro')
    plt.title("%s"%name)
    plt.xlim(tmin,tmax)
    locs,labels = plt.xticks()
    plt.xticks(locs, map(lambda x: "%g" % x, locs))
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%g" % x, locs))
    plt.axvline(x=lastsun,ls='--', color='black') # last Monday
    plt.xlabel("JD-2450000")

    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue())
    html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
    img.close()
    plt.close(fig)


    return html_code

def gen_followup_multi(name, data, ndata, data_bands, tmin, tmax):
    x = data[0][0][:]
    y = data[0][1][:]
    err = data[0][2][:]
    fig=plt.figure(figsize=(8,4)) # figsize is in inches
    plt.gca().invert_yaxis()
    plt.title("%s"%name)
    plt.xlim(tmin,tmax)
    plt.errorbar(x,y,yerr=err,fmt='ro')

    bands = ['u','B','g','V','B2pg',
             'r','R','R1pg','i','I',
             'Ipg','z', 'B1pg', 'G', 'H',
             'K', 'J', 'ZTF_g', 'ZTF_r', 'ZTF_i',
             'ZTF_al_g', 'ZTF_al_r']
    clist = ['dodgerblue', 'skyblue', 'green', 'orange',  'slategrey',
             'darkred', 'salmon', 'indianred', 'dimgrey', 'peru',
             'darkslateblue', 'hotpink', 'lightseagreen', 'navy', 'dimgrey',
             'tan', 'indigo', 'teal', 'crimson', 'blueviolet',
             'darkslategrey', 'palevioletred']
    #maxMag, minMag = 25., 0.
    print(data_bands)
    for i in range(len(data_bands)):
        b = data_bands[i]
        idx = 0
        for k in range(len(bands)):
            if(bands[k]==b):
                idx = k
                continue
        if(idx>0):
            print (b, clist[idx], bands[idx])
            ftime = data[i][0][:]
            fmags = data[i][1][:]
            ferr = data[i][2][:]
            #maxMag = min(min(fmags), maxMag)
            #minMag = max(max(fmags), minMag)
            plt.errorbar(ftime,fmags,yerr=ferr,marker='.', color=clist[idx], ls='', label=b)
    #plt.ylim(minMag, maxMag)
    locs,labels = plt.xticks()
    plt.xticks(locs, map(lambda x: "%g" % x, locs))
    locs,labels = plt.yticks()
    plt.yticks(locs, map(lambda x: "%g" % x, locs))
    plt.axvline(x=lastsun,ls='--', color='black') # last Monday
    plt.legend(loc='best', ncol=2)
    plt.xlabel("JD-2450000")

    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue())
    html_code = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
    img.close()
    plt.close(fig)

    return html_code
#------------###-------------#---------------#---------#

def plot_ulens(name, photArray, GaiaZeroPoint):

    ##### Microlensing model
    #times, mags, label = alert_single_img.average_alertlightcurve(alertlightcurveArray, GaiaZeroPoint)
    nrtrid, times, mags = phot_from_array(photArray)
    times =  times + mjdoffset - 2450000.
    err = np.full(len(times), 0.02) # fixed value of error, please change if needed
    eta=1./(nrtrid-1.)/np.var(mags)*np.sum((mags[1:]-mags[:-1])*(mags[1:]-mags[:-1]))
    #skewness
    skew=scipy.stats.skew(mags)
    #amplitude
    ampl = np.amax(mags) - np.amin(mags)
    fitparams, chi2dof = fit_ulensfixedbl(times, mags, err)
    ulensuri = gen_modellightcurve(name, times, mags, fitparams, GaiaZeroPoint)

    return ulensuri, chi2dof, fitparams

def plot_ulens_par(name, photArray, GaiaZeroPoint, alpha, delta):
    ##### Microlensing model
    #times, mags, label = alert_single_img.average_alertlightcurve(alertlightcurveArray, GaiaZeroPoint)
    nrtrid, times, mags = phot_from_array(photArray)
    times =  times + mjdoffset - 2450000.
    err = np.full(len(times), 0.02) # fixed value of error, please change if needed
    eta=1./(nrtrid-1.)/np.var(mags)*np.sum((mags[1:]-mags[:-1])*(mags[1:]-mags[:-1]))
    #skewness
    skew=scipy.stats.skew(mags)
    #amplitude
    ampl = np.amax(mags) - np.amin(mags)
    fitparams1, chi2dof1 = fit_ulensfixedbl(times, mags, err)
    fitparams2, chi2dof2 = fit_ulensparallax(times, mags, err, alpha, delta, fitparams1)
    ulensuri = gen_modellightcurve_par(name, times, mags, fitparams1, fitparams2, alpha, delta, GaiaZeroPoint)

    return ulensuri, chi2dof1, fitparams1, chi2dof2, fitparams2

def plot_ulens_par_fixedbl(name, times, mags, GaiaZeroPoint, alpha, delta):
    ##### Microlensing model
    times =  times - 2450000.
    nrtrid = len(times)
    err = getGaiaErrors(mags)
    tLast = times[-1]

    #check for followup:
    datajson, followup = get_followup(name)
    ztffol, ztf_data, ztf_bands = getLightCurveZTF(name, alpha, delta)
    ztf_al_fol = 0.
    ztf_al_fol, ztf_al_data, ztf_al_bands = getLightCurveZTFAlert(alpha, delta)
    datasets = []

    ndata = 1
    datasets.append((times, mags, err))
    data_bands = ['Gaia']
    if (followup != 0):
        mjd0=np.array(datajson['mjd'])
        mag0=np.array(datajson['mag'])
        magerr0=np.array(datajson['magerr'])
        filter0=np.array(datajson['filter'])
        caliberr0=np.array(datajson['caliberr'])
        obs0 = np.array(datajson['observatory'])
        filter_list = ['u','B','g','V','B2pg','r','R','R1pg','i','I','Ipg','z', 'B1pg', 'G', 'H', 'K', 'J']
        for band in filter_list:
            indexes = np.where(filter0 == band)
            ftime = mjd0[indexes] + jdoffset2 - 2450000.
            fmags = mag0[indexes]
            ferr = magerr0[indexes]
            indexes2 = np.where(ferr != -1.0)
            ftime = ftime[indexes2]
            fmags = fmags[indexes2]
            ferr = ferr[indexes2]
            ft = []
            fm = []
            fe = []
            for i in range(len(ftime)):
                ft.append(float(ftime[i]))
                fm.append(float(fmags[i]))
                fe.append(float(ferr[i]))
            if(len(ft)>1):
                if(ft[-1]>tLast):
                    tLast = ft[-1]
                ndata = ndata + 1
                datasets.append((ft,fm, fe))
                data_bands.append(band)
                #print(band, len(ft))
            # print '<br> Dataset: ', ft

    if(ztffol>0):
        b = 0
        for d in ztf_data:
            t = d[:,0]-2450000.
            m = d[:,1]
            e = d[:,2]
            if(t[-1]> tLast):
                tLast = t[-1]
            ndata = ndata + 1
            datasets.append((t,m,e))
            data_bands.append(ztf_bands[b])
            b = b + 1


    if(ztf_al_fol>0):
        for i in range(2):
            if(len(ztf_al_data[i][0][:])>1):
                if(ztf_al_data[i][0][-1]>tLast):
                    tLast = ztf_al_data[i][0][-1]
                ndata = ndata + 1
                #print ztf_al_data[i][0][:], ztf_al_data[i][1][:]
                datasets.append((ztf_al_data[i][0][:], ztf_al_data[i][1][:], ztf_al_data[i][2][:]))
                data_bands.append(ztf_al_bands[i])

    # if(ndata == 1):
        # fitparams1, chi2dof1 = fit_ulensfixedbl_single(times, mags, err)
        # fitparams2, chi2dof2 = fit_ulensparallax_fixedbl_single(times, mags, err, alpha, delta,fitparams1)
        # ulensuri = gen_modellightcurve_par_fixedbl_single(name, times, mags, fitparams1, fitparams2, alpha, delta, GaiaZeroPoint)
        # followupuri = gen_followup_single(times, mags, err)
    # else:
        # fitparams1, chi2dof1 = fit_ulensfixedbl_multi(datasets, ndata)
        # fitparams2, chi2dof2 = fit_ulensparallax_fixedbl_multi(datasets, ndata, alpha, delta,fitparams1)
        # ulensuri = gen_modellightcurve_par_fixedbl_multi(name, datasets, ndata, fitparams1, fitparams2, alpha, delta, GaiaZeroPoint)
        # followupuri =  gen_followup_multi(name, datasets, ndata)
    fitparams1, chi2dof1 = fit_ulensfixedbl_single(times, mags, err)
    fitparams2, chi2dof2 = fit_ulensparallax_fixedbl_single(times, mags, err, alpha, delta,fitparams1)
    #print(times[0]-70, tLast+70)
    ulensuri = gen_modellightcurve_par_fixedbl_single(name, datasets, fitparams1, fitparams2, alpha, delta, GaiaZeroPoint, times[0]-70, tLast+70)
    if(ndata == 1):
        tmax = tLast+70.
        #tmin = tLast - 250
        tmin = times[0]-70.
        followupuri1 = gen_followup_single(name,times, mags, err, tmin,tmax)
        tmax = tLast+30.
        tmin = times[-1]-200.
        followupuri2 = gen_followup_single(name,times, mags, err, tmin,tmax)
    else:
        tmax = tLast+70.
        #tmin = tLast - 250
        tmin = times[0]-70.
        followupuri1 =  gen_followup_multi(name, datasets, ndata, data_bands, tmin, tmax)
        tmax = tLast+30.
        tmin = times[-1]-200.
        followupuri2 =  gen_followup_multi(name, datasets, ndata, data_bands, tmin, tmax)
    return ulensuri, followupuri1, followupuri2, chi2dof1, fitparams1, chi2dof2, fitparams2

def phot_from_array(photArray):
    times = []
    mags = []
    t = []
    m = []
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    m5 = []
    m6 = []
    m7 = []
    m8 = []
    m9 = []
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    t6 = []
    t7 = []
    t8 = []
    t9 = []
    for p in photArray:
        time0=p[0]-mjdoffset #MJD
        af1=p[1]
        af2=p[2]
        af3=p[3]
        af4=p[4]
        af5=p[5]
        af6=p[6]
        af7=p[7]
        af8=p[8]
        af9=p[9]
        aferr1=p[10]
        aferr2=p[11]
        aferr3=p[12]
        aferr4=p[13]
        aferr5=p[14]
        aferr6=p[15]
        aferr7=p[16]
        aferr8=p[17]
        aferr9=p[18]
        mean=p[19]
        transitid=p[20]
        ttransit=(((transitid >> 17) & OBT_MASK) * OBT_FACTOR) /1e9 / (60.*60*24)

        tt=time0

        t.append(tt)
        ccdscantime=4.4/60./60.
        t1.append(tt+1*ccdscantime)
        t2.append(tt+2*ccdscantime)
        t3.append(tt+3*ccdscantime)
        t4.append(tt+4*ccdscantime)
        t5.append(tt+5*ccdscantime)
        t6.append(tt+6*ccdscantime)
        t7.append(tt+7*ccdscantime)
        t8.append(tt+8*ccdscantime)
        t9.append(tt+9*ccdscantime)
        m1.append(af1)
        m2.append(af2)
        m3.append(af3)
        m4.append(af4)
        m5.append(af5)
        m6.append(af6)
        m7.append(af7)
        m8.append(af8)
        m9.append(af9)
        
    #sorting time array
    zipped=0
    zipped = zip(t1, m1)
    zipped.sort()
    st1=[i for (i, s) in zipped]
    sf1=[s for (i, s) in zipped]
    zipped=0
    zipped = zip(t2, m2)
    zipped.sort()
    st2=[i for (i, s) in zipped]
    sf2=[s for (i, s) in zipped]
    zipped = zip(t3, m3)
    zipped.sort()
    st3=[i for (i, s) in zipped]
    sf3=[s for (i, s) in zipped]
    zipped = zip(t4, m4)
    zipped.sort()
    st4=[i for (i, s) in zipped]
    sf4=[s for (i, s) in zipped]
    zipped = zip(t5, m5)
    zipped.sort()
    st5=[i for (i, s) in zipped]
    sf5=[s for (i, s) in zipped]
    zipped = zip(t6, m6)
    zipped.sort()
    st6=[i for (i, s) in zipped]
    sf6=[s for (i, s) in zipped]
    zipped = zip(t7, m7)
    zipped.sort()
    st7=[i for (i, s) in zipped]
    sf7=[s for (i, s) in zipped]
    zipped = zip(t8, m8)
    zipped.sort()
    st8=[i for (i, s) in zipped]
    sf8=[s for (i, s) in zipped]
    zipped = zip(t9, m9)
    zipped.sort()
    st9=[i for (i, s) in zipped]
    sf9=[s for (i, s) in zipped]

    times = t
    for i in range(len(t)):
        fl = np.nanmedian([sf1[i], sf2[i], sf3[i], sf4[i], sf5[i], sf6[i], sf7[i], sf8[i], sf9[i]])
        mags.append(fl)
    #print mags
    #von neumann
    nrtrid = len(times)
    times = np.array(times)
    mags = np.array(mags)
    mags0 = np.array(mags)
    mags = mags[~np.isnan(mags0)]
    times = times[~np.isnan(mags0)]
    return nrtrid, times, mags

# using ZKR's code getfollowup.py as a base here
def get_followup(name):
    followup=1
    data1 = 0.
    try:
        print("Opening followup page")
        br = mechanize.Browser()
        followuppage=br.open('http://gsaweb.ast.cam.ac.uk/followup/')
        req=br.click_link(text='Login')
        br.open(req)
        br.select_form(nr=0)
        br.form['hashtag']='ZKR_ceb5e70b2c4e9b8866d7d62b9c60f811'
        br.submit()
        #print "Logged in!"
        #print "Requesting followup!"
        r=br.open('http://gsaweb.ast.cam.ac.uk/followup/get_alert_lc_data?alert_name=ivo:%%2F%%2F%s'%name)
        page=br.open('http://gsaweb.ast.cam.ac.uk/followup/get_alert_lc_data?alert_name=ivo:%%2F%%2F%s'%name)
        pagetext=page.read()
        data1=json.loads(pagetext)
        #print "Followup downloaded. Proceeding further"
        filter_list = ['u','B','g','V','B2pg','r','R','R1pg','i','I','Ipg','z', 'B1pg', 'G', 'H', 'K', 'J']
        if len(set(data1["filter"]) & {'u', 'B', 'g', 'V', 'B2pg', 'r', 'R', 'R1pg', 'i', 'I', 'Ipg', 'z', 'B1pg', 'G',
									   'H', 'K', 'J'})>0:
            fup=[data1["mjd"],data1["mag"],data1["magerr"],data1["filter"],data1["observatory"]]
            print("Followup data downloaded!")
        else:
            followup = 0.
            print("No followup available.")
    except mechanize.HTTPError as e:
        followup = 0.
        print(e)

    #displaying array of data as json if no arguments

    return data1, followup

def getLightCurveGaia(name):
    url = ("http://gsaweb.ast.cam.ac.uk/alerts/alert/%s/lightcurve.csv")%(name)
    req = requests.get(url)
    text = req.text
    lines = text.split('\n')
    gaiahjd = []
    gaiamag = []
    #
    for l in lines:
        col = l.split(',')
        if (len(col)>1):
            if (len(col)==3 and (col[1] != 'JD(TCB)')):
                if (col[2] != 'null' and  col[2] != 'untrusted'):
                    gaiahjd.append(float(col[1]))
                    gaiamag.append(float(col[2]))
    return gaiahjd, gaiamag

def getLightCurveZTF(name, ra, dec):
    rad = 2./60./60.
    ztffol = 0
    ztf_data = []
    ztf_bands = []

    bands = ['g', 'r', 'i']

    try:
        for band in bands:
            data = []
            url = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?POS=CIRCLE+%f+%f+%f&BANDNAME=%s&NOBS_MIN=3&BAD_CATFLAGS_MASK=32768&FORMAT=ipac_table"%(ra, dec, rad, band)
            print(url)
            req = requests.get(url)
            text = req.text
            lines = text.split('\n')

            i = 0
            for line in lines:
                col = line.split()
                #print(len(col), col)
                if(len(col)>1):
                    if(col[0][0] != "|" and col[0][0] != "\\" and col[0][0] != "<"):
                        # print(i, col[2], col[4], col[5], col[7])
                        data.append((float(col[2]), float(col[4]), float(col[5])))
                i += 1
            data = np.asarray(data)
            if(len(data)>0):
                ztffol = ztffol + 1
                ztf_data.append(data)
                ztf_bands.append("ZTF_"+band)

    except requests.HTTPError as exception:
        print(exception)

    if(ztffol>0):
        print("Data from ZTF obtained!")
    else:
        print("No ZTF data available.")

    return ztffol, ztf_data, ztf_bands

    # old below
# def getLightCurveZTF(name, ra, dec, wsdbcur):
    # radius = 2./60./60.
    # ztffol = 0
    # ztf_data = []
    # ztf_bands = []

    # if(wsdbcur != None):
        # for band in ['g', 'r', 'i']:
            # sql_string = 'select mjd_%s, mag_%s, magerr_%s from ztf_dr3.lc_gri where q3c_radial_query(ra, dec, %f, %f, %f);'%(band, band, band, ra, dec, radius)
            # wsdbcur.execute(sql_string)
            # results = wsdbcur.fetchall()
            # results = np.array([results])

            # if(len(results.shape)>3):
                # #print(results.shape, len(results.shape))
                # results = np.array([results[0][0]])
                # ztffol = ztffol + 1
                # #print(results.shape, len(results.shape))
                # ztf_mjd = np.array(results[0][0][:])- 50000.
                # ztf_mags = np.array(results[0][1][:])
                # ztf_errs = np.array(results[0][2][:])
                # #print(len(ztf_mjd), len(ztf_mags), len(ztf_errs))
                # ztf_data.append((ztf_mjd, ztf_mags, ztf_errs))
                # ztf_bands.append("ZTF_"+band)
    # if(ztffol>0):
        # print("Data from ZTF obtained!")
    # else:
        # print("No ZTF data available.")

    # return ztffol, ztf_data, ztf_bands
    # old below
    # ztf_list = np.genfromtxt("ztf_dr1_list.dat", dtype='str', unpack=True)
    # ztf_name = "ztf_"+name+".dat"
    # idx = np.where(ztf_list == ztf_name)
    # if(idx[0]):
        # time, mag, err = np.genfromtxt(ztf_name, dtype=np.float, usecols=(0,1,2), unpack=True)
        # return time-2450000., mag, err
    # else:
        # return np.array([0]),np.array([0]),np.array([0])

# Get data from ZTF alerts
#ZTF MARS:
def getLightCurveZTFAlert(ra, dec):
    mars_time, mars_mag, mars_err, mars_filter = getztfmars(ra, dec)
    marsmjd_g = []
    marsmag_g = []
    marserr_g = []
    marsmjd_r = []
    marsmag_r = []
    marserr_r = []
    fol = 0.

    #print mars_filter
    if(len(mars_time) > 1):
        fol = 1
        for i in range(len(mars_time)):
            if (mars_filter[i] =='g') :
                #print mars_filter[i] , mars_mag[i]
                if (mars_mag[i] is not None):
                    marsmjd_g.append(float(mars_time[i])-2450000.5)
                    marsmag_g.append(float(mars_mag[i]))
                    marserr_g.append(float(mars_err[i]))
                    #print "Elo!"
                    #if (printdata==1): print("{} {} {} {}<br>".format(float(mars_time[i])-2400000.5, float(mars_mag[i]), float(mars_err[i]), mars_filter[i]))
            if (mars_filter[i] =='r') :
                #print mars_filter[i], mars_mag[i]
                if (mars_mag[i] is not None):
                    marsmjd_r.append(float(mars_time[i])-2450000.5)
                    marsmag_r.append(float(mars_mag[i]))
                    marserr_r.append(float(mars_err[i]))
                    #print "Elo2!"
                    #if (printdata==1): print("{} {} {} {}<br>".format(float(mars_time[i])-2400000.5, float(mars_mag[i]), float(mars_err[i]), mars_filter[i]))
    marsmjd_g = np.asarray(marsmjd_g)
    marsmag_g = np.asarray(marsmag_g)
    marserrd_g = np.asarray(marserr_g)
    marsmjd_r = np.asarray(marsmjd_r)
    marsmag_r = np.asarray(marsmag_r)
    marserrd_r = np.asarray(marserr_r)

    #print mars_time, marsmjd_g, marsmjd_r
    ztf_data = []
    ztf_bands = []
    ztf_data.append((marsmjd_g, marsmag_g, marserr_g))
    ztf_bands.append("ZTF_al_g")

    ztf_data.append((marsmjd_r, marsmag_r, marserr_r))
    ztf_bands.append("ZTF_al_r")

    if(fol>0):
        print("Data from ZTF Alerts obtained!")
    else:
        print("No ZTF Alerts data available.")

    return fol, ztf_data, ztf_bands

# Wyrzykowski-Rybicki relation for Gaia errors ;)
# From K. Rybicki's code "gaia_err.py"
def getGaiaErrors(mag):
    a1 = 0.2
    b1 = -5.2
    a2 = 0.26
    b2 = -6.26

    err = []
    for i in range(0,len(mag)):
        if mag[i] <= 13.5:
             err_corr=a1*13.5+b1
        elif mag[i] > 13.5 and mag[i] <= 17.:
            err_corr=a1*mag[i]+b1
        elif mag[i] > 17.:
            err_corr=a2*mag[i]+b2
        else:
            err_corr = np.NaN
        err.append(10.**err_corr)
    return err
