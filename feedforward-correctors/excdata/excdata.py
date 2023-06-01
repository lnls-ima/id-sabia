#!/usr/bin/env python-sirius


import numpy as np
import matplotlib.pyplot as plt

# https://github.com/lnls-fac/lnls
from lnls.rotcoil import RotCoilMeas_SIFCH


COMMENTS = [
        ('data was fit with a polynomial of order 4 and '
         'remanent/ambient field subtracted.'),
        ('data in https://github.com/lnls-ima/id-sabia/tree/master/'
         'feedforward-correctors/model-03/measurement/magnetic/rotcoil'),
        ('script https://github.com/lnls-ima/id-sabia/tree/master/'
         'feedforward-correctors/excdata/excdata.py'),
    ]

# use SIFCH as template, with adaptations
RotCoilMeas_SIFCH.lnls_ima_path = '/home/ximenes/repos-dev/'
RotCoilMeas_SIFCH.magnet_type_name = 'id-sabia/feedforward-correctors/'
RotCoilMeas_SIFCH.model_version = 'model-03/'
RotCoilMeas_SIFCH.magnet_type_label = 'FFC'
RotCoilMeas_SIFCH.excitation_type = ''


def create_excdata(serials, exc_type, currs, harms, fit_order):
    """."""
    conv_mpoles_sign = {'ch':+1, 'cv':+1, 'qs':-1}[exc_type]
    main_harmonic_type = {'ch': 'normal', 'cv': 'skew', 'qs': 'skew'}[exc_type]
    main_harmonic = {'ch': 1, 'cv': 1, 'qs': 2}[exc_type]
    main_harmonic_idx = harms.index(main_harmonic)
    excdata_label = {
        'ch': 'si-corrector-epu50-ch',
        'cv': 'si-corrector-epu50-cv',
        'qs':'si-corrector-epu50-qs'}[exc_type]

    # retrieve rotcoil data
    rcd = dict()
    for serial in serials:
        RotCoilMeas_SIFCH.conv_mpoles_sign = conv_mpoles_sign
        rcd[serial] = RotCoilMeas_SIFCH(serial_number=serial)

    # get multipoles in numpy array (nr_currents x nr_harmonics)
    currents = intmpole_norm = intmpole_skew = None
    for serial, rc in rcd.items():
        intmpole_norm_ = np.zeros((rc.size, len(harms)))
        intmpole_skew_ = np.zeros((rc.size, len(harms)))
        currents_ = np.array(rc.get_currents(exc_type))
        for i, h in enumerate(harms):
            intmpole_norm_[:, i] = rc.get_intmpole_normal_avg(exc_type, h)
            intmpole_skew_[:, i] = rc.get_intmpole_skew_avg(exc_type, h)
        if currents is None:
            currents = currents_
            intmpole_norm = intmpole_norm_
            intmpole_skew = intmpole_skew_
        else:
            currents = np.hstack((currents, currents_))
            intmpole_norm = np.vstack((intmpole_norm, intmpole_norm_))
            intmpole_skew = np.vstack((intmpole_skew, intmpole_skew_))

    # fit polynomials
    intmpole_norm_fit = np.zeros((len(currs), len(harms)))
    intmpole_skew_fit = np.zeros((len(currs), len(harms)))
    intmpole_norm_fit_at_zero = []
    intmpole_skew_fit_at_zero = []
    for i in range(len(harms)):
        # norm
        pfit = np.polyfit(currents, intmpole_norm[:, i], fit_order)
        intmpole_fit_at_zero_ = np.polyval(pfit, 0)
        intmpole_norm_fit_at_zero.append(intmpole_fit_at_zero_)
        intmpole_fit_ = np.polyval(pfit, currs) - intmpole_fit_at_zero_
        intmpole_norm_fit[:, i] = intmpole_fit_
        # skew
        pfit = np.polyfit(currents, intmpole_skew[:, i], fit_order)
        intmpole_fit_at_zero_ = np.polyval(pfit, 0)
        intmpole_skew_fit_at_zero.append(intmpole_fit_at_zero_)
        intmpole_fit_ = np.polyval(pfit, currs) - intmpole_fit_at_zero_
        intmpole_skew_fit[:, i] = intmpole_fit_

    # save excdata file
    txt = RotCoilMeas_SIFCH.get_excdata_text(
        pwrsupply_polarity='',
        magnet_type_label='FFC',
        magnet_serial_number=','.join([str(serial) for serial in serials]),
        data_set=exc_type.upper(),
        main_harmonic=main_harmonic,
        main_harmonic_type=main_harmonic_type,
        harmonics=harms,
        currents=currs,
        mpoles_n=intmpole_norm_fit,
        mpoles_s=intmpole_skew_fit,
        filename=excdata_label,
        comments=COMMENTS
    )
    with open(excdata_label + '.txt', 'w') as fp:
        for line in txt:
            fp.write(line + '\n')

    # plot
    tesla_2_gauss = 1e4
    meter_2_cm = 1e2
    tesla_meter_2_gauss_cm = meter_2_cm * tesla_2_gauss
    labels = {
        'ch': (
                'Integrated Normal Dipolar Field [G.cm]',
                'Main Multipole - CH (shifted)', tesla_meter_2_gauss_cm),
        'cv': (
                'Integrated Skew Dipolar Field [G.cm]',
                'Main Multipole - CV (shifted)', tesla_meter_2_gauss_cm),
        'qs': (
                'Integrated Skew Quadrupolar Field [G]',
                'Main Multipole - QS (shifted)', tesla_2_gauss),
        }
    labely, title, coeff = labels[exc_type]
    
    for serial, rc in rcd.items():

        currents_ = rc.get_currents(exc_type)
        if exc_type == 'ch':
            intmpole = np.array(
                rc.get_intmpole_normal_avg(exc_type, main_harmonic))
            intmpole_fit_at_zero = intmpole_norm_fit_at_zero
            intmpole_fit = intmpole_norm_fit
        else:
            intmpole = np.array(
                rc.get_intmpole_skew_avg(exc_type, main_harmonic))
            intmpole_fit_at_zero = intmpole_skew_fit_at_zero
            intmpole_fit = intmpole_skew_fit

        intmpole_fit_at_zero_main = intmpole_fit_at_zero[main_harmonic_idx]
        intmpole2 = intmpole - intmpole_fit_at_zero_main    
        plt.plot(currents_, coeff*intmpole2, 'o', label='FFC-' + serial)

    idx_fit = harms.index(main_harmonic)
    plt.plot(
        currs, coeff*intmpole_fit[:, idx_fit], '.-', label='quadratic fit')

    plt.xlabel('Current [A]')
    plt.ylabel(labely)
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(excdata_label + '.png')
    plt.show()

    
def run():
    
    serials = ['01', '02']
    currs = np.linspace(-10, 10, 11)
    fit_order = 4

    exc_type = 'ch'
    harms = [0+1, 2+1, 4+1]
    create_excdata(serials, exc_type, currs, harms, fit_order)

    exc_type = 'cv'
    harms = [0+1, 2+1, 4+1]
    create_excdata(serials, exc_type, currs, harms, fit_order)

    exc_type = 'qs'
    harms = [1+1, 5+1, 9+1]
    create_excdata(serials, exc_type, currs, harms, fit_order)


if __name__ == "__main__":
    run()
