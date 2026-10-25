# Historical inference compatibility
This directory preserves the originally shared architecture, configuration and checkpoint. It is not MST-Net V3 and must not be used to reproduce the revised LAI comparison. Its serialized checkpoint may require scikit-learn 1.3.2. Load only the trusted supplied file.

The historical biomass analyses retain their supplied frozen LAI inputs. The presence of this legacy checkpoint does not establish that it generated every retained LAI value. Do not replace those inputs automatically with new V3 predictions.

The supplied legacy code, configuration and checkpoint are covered by the MIT License in the package root. Third-party dependencies retain their own licenses.
