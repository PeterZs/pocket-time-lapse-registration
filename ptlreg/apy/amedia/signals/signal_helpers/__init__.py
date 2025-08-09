import os;
from ptlreg.apy import *
from ptlreg.apy.amedia.media import *
from ptlreg.apy.amedia.media.Image import ImageMethod

def show_signal_with_spectrum(signal, title=None, time_range = None, frequency_range=None):
    """
    Display the time signal next to its power spectrum
    :param signal:
    :param title:
    :param time_range:
    :param frequency_range:
    """
    f, ax = plt.subplots(2, 1  )# sharex=True
    axes =[ax[0], ax[1]];
    f.set_size_inches([9 ,4])
    plt.tight_layout()

    # if(frequency_range is None):
    #     frequency_range = default_frequency_range;

    if(title is not None):
        axis_titles = ['{} Signal'.format(title), '{} Spec'.format(title)];
        f.number = title;
    else:
        axis_titles = ['Signal', 'Spec'];

    signal.plot(axis=axes[0], time_range=time_range);
    axes[0].set_title(axis_titles[0]);

    signal.plot_amplitude_spectrum(axis=axes[1], frequency_range=frequency_range);
    axes[1].set_title(axis_titles[1]);


def compare_signals_with_spectra(signals_dict, time_range=None, frequency_range=None):
    """

    :param signals_dict:
    :param time_range:
    :param frequency_range:
    """
    f, ax = plt.subplots(2, 1)  # sharex=True
    axes = [ax[0], ax[1]]
    f.set_size_inches([9, 4])
    plt.tight_layout()
    axis_titles = ["Signal", "Spec"]

    #     legend = list(map(lambda x: x.title, signals))

    signals = [];
    signal_labels = [];
    for skey, svalue in signals_dict.items():
        signals.append(svalue)
        signal_labels.append(skey)

    for sgn in signals:
        sgn.plot(axis=axes[0], time_range=time_range)
    axes[0].set_title(axis_titles[0])
    axes[0].legend(signal_labels)

    for sgn in signals:
        sgn.plot_power_spectrum(axis=axes[1], frequency_range=frequency_range)
    axes[1].set_title(axis_titles[1])
    axes[1].legend(signal_labels)