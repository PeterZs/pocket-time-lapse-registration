# __all__ = ['ColorScheme'];

import palettable.colorbrewer.qualitative
import palettable.scientific.sequential
import palettable.colorbrewer.sequential
import palettable.colorbrewer.diverging


class Palette(object):
    def __init__(self, palettable, **kwargs):
        self.palettable=palettable;

    def __call__(self, normalized_value):
        return self.palettable.mpl_colormap(normalized_value);

    def getContinuousColor(self, val):
        return self.palettable.mpl_colormap(val);

    @property
    def colors(self):
        return self.palettable.colors;


    def __getitem__(self, key):
        return self.palettable.colors.__getitem__(key);

    def __iter__(self):
        return self.palettable.colors.__iter__();


class ColorScheme(object):
    DivergingSchemes= palettable.colorbrewer.diverging;
    SequentialSchemes = palettable.scientific.sequential;
    QualitativeSchemes = palettable.colorbrewer.qualitative;

    # DivergingSchemes=dict(
    #     Earth_7=Earth_7,
    #     Fall_7=Fall_7,
    #     Geyser_7=Geyser_7,
    #     Temps_7 = Temps_7,
    #     TealRose_7=TealRose_7,
    #     Tropic_7=Tropic_7,
    #     Spectral_11=Spectral_11
    # )

    # QualitativeSchemes=dict(
    #     Antique_10=Antique_10,
    #     Bold_10=Bold_10,
    #     Pastel_10=Pastel_10,
    #     Prism_10=Prism_10,
    #     Safe_10=Safe_10,
    #     Vivid_10=Vivid_10
    # )
    # SequentialSchemes=dict(
    #     BluGrn_7=BluGrn_7,
    #     BluYl_7=BluYl_7,
    #     BrwnYl_7=BrwnYl_7,
    #     Burg_7=Burg_7
    # )

    Qualitative = palettable.colorbrewer.qualitative.Paired_12;
    Sequential = palettable.scientific.sequential.Imola_20;
        # BuGn_9;
        # Imola_20;
        # YlGnBu_9;
    Diverging = palettable.colorbrewer.diverging.Spectral_11;



    @classmethod
    def GetSequential(cls):
        return Palette(cls.Sequential);

    @classmethod
    def GetDiverging(cls):
        return Palette(cls.Diverging);

    @classmethod
    def GetQualitative(cls):
        return Palette(cls.Qualitative);



    @classmethod
    def showContinuous(cls, scheme=None):
        if(scheme is None):
            scheme = 'Sequential';
        if(scheme.lower()=='qualitative'):
            cls.Qualitative.show_continuous_image();
        elif(scheme.lower()=='sequential'):
            cls.Sequential.show_continuous_image();
        elif(scheme.lower()=='diverging'):
            cls.Diverging.show_continuous_image();

    @classmethod
    def showDiscrete(cls, scheme=None):
        if(scheme is None):
            scheme = 'Qualitative';
        if(scheme.lower()=='qualitative'):
            cls.Qualitative.show_discrete_image();
        elif(scheme.lower()=='sequential'):
            cls.Sequential.show_discrete_image();
        elif(scheme.lower()=='diverging'):
            cls.Diverging.show_discrete_image();

