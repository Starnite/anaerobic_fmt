import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import skbio
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pylab
import os
from collections import OrderedDict

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.io import output_file

def unique(seq, idfun=None):
# order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        if item != np.nan:
            marker = idfun(item)
            if marker in seen: continue
            seen[marker] = 1
            result.append(item)
    return result
    #takes in more params
def engraftment_capability_all_otus_all_timepoints(otu_table_counts, otu_table_abundance, metadata):
    engraftment_capablity_df = pd.DataFrame(columns=["timepoint","patient","otu", "otu_source","count", "abundance", "aerobic", "donor", "donor_count", "donor_abundance", "donor_direct"])
    donor_only = {} #dict where keys are patient ids and values are lists of indices representing seq vars that are in
                    #the corresponding donor but not in the patient originally, where the first list is direct and the second pma
    for i in otu_table_counts.index:
        if metadata.loc[i, "timepoint"] == "Pre-FMT":
            donor_only[metadata.loc[i, "person_id"]] = [list(set(list(otu_table_counts.loc[metadata.loc[i, "direct_donor_fmt_id"]].nonzero()[0]))-set(list(otu_table_counts.loc[i].nonzero()[0]))),list(set(list(otu_table_counts.loc[metadata.loc[i, "pma_donor_fmt_id"]].nonzero()[0]))-set(list(otu_table_counts.loc[i].nonzero()[0])))]
    both = {}
    for i in otu_table_counts.index:
        if metadata.loc[i, "timepoint"] == "Pre-FMT":
            both[metadata.loc[i, "person_id"]] = [list(set(list(otu_table_counts.loc[metadata.loc[i, "direct_donor_fmt_id"]].nonzero()[0]))&set(list(otu_table_counts.loc[i].nonzero()[0]))),list(set(list(otu_table_counts.loc[metadata.loc[i, "pma_donor_fmt_id"]].nonzero()[0]))&set(list(otu_table_counts.loc[i].nonzero()[0])))]
    patient_only = {}
    for i in otu_table_counts.index:
        if metadata.loc[i, "timepoint"] == "Pre-FMT":
            patient_only[metadata.loc[i, "person_id"]] = [list(set(list(otu_table_counts.loc[i].nonzero()[0]))-set(list(otu_table_counts.loc[metadata.loc[i, "direct_donor_fmt_id"]].nonzero()[0]))),list(set(list(otu_table_counts.loc[i].nonzero()[0]))-set(list(otu_table_counts.loc[metadata.loc[i, "pma_donor_fmt_id"]].nonzero()[0])))]
    timepoints = list(metadata["timepoint"].unique())[:-1]
    #timepoints = timepoints[~np.isnan(timepoints)]
    patients = list(metadata["patient_num"].unique())[:-1]
    #patients = patients[~np.isnan(patients)]
    otus = otu_table_counts.columns

    for timepoint in timepoints:
        for patient in patients:
            try:
                sample_id = metadata.index[(metadata["person_id"] == patient) & (metadata["timepoint"] == timepoint)][0]

                aerobic = metadata.loc[sample_id, "anaerobic_fmt"]
                donor = int(metadata.loc[sample_id,"donor_fmt_num"])
                for otu in otus:
                    count = otu_table_counts.loc[sample_id, otu]
                    abundance = otu_table_abundance.loc[sample_id, otu]
                    otu_index = otu_table_counts.columns.get_loc(otu)
                    for i, dp in enumerate(["direct", "pma"]):

                        if otu_index in donor_only[patient][i]:
                            otu_source = "donor"
                        elif otu_index in both[patient][i]:
                            otu_source = "both"
                        elif otu_index in patient_only[patient][i]:
                            otu_source = "patient"
                        else:
                            otu_source = "environmental"
                        donor_count = otu_table_counts.loc[metadata.loc[sample_id, "{}_donor_fmt_id".format(dp)], otu]
                        donor_abundance = otu_table_abundance.loc[metadata.loc[sample_id, "{}_donor_fmt_id".format(dp)], otu]
                        engraftment_capablity_df = engraftment_capablity_df.append({"timepoint":timepoint, "patient":patient, "otu":otu, "otu_source":otu_source, "count":count, "abundance":abundance, "aerobic":aerobic, "donor":donor, "donor_count":donor_count, "donor_abundance":donor_abundance, "donor_direct":dp,}, ignore_index=True)
            except Exception as e:
                print(e)
                print(patient, timepoint)
                pass
    return engraftment_capablity_df

genus_percent = pd.read_csv("abundances_by_tax_level_relfreq/Genus_otu_table.csv", index_col=0).transpose().drop(["ndc432", "ndc442"])
genus_counts = pd.read_csv("abundances_by_tax_level_counts/Genus_otu_table.csv", index_col=0).transpose().drop(["ndc432", "ndc442"])
metadata= pd.read_csv("sample_metadata.csv", index_col=0)
genus_engraftment = engraftment_capability_all_otus_all_timepoints(genus_counts, genus_percent, metadata)
genus_engraftment["count"] = genus_engraftment["count"] + 1
genus_engraftment["donor_count"] = genus_engraftment["donor_count"] + 1
genus_engraftment["abundance"] = genus_engraftment["abundance"] + 0.000001
genus_engraftment["donor_abundance"] = genus_engraftment["donor_abundance"] + 0.000001
genus_engraftment_pma = genus_engraftment[genus_engraftment["donor_direct"]=="pma"]
genus_engraftment_direct = genus_engraftment[genus_engraftment["donor_direct"]=="direct"]
def lfc_all_timepoints_all_donors(all_engraftment):
    """
    all_engraftment: abundance info for everything
    """
    lfc_df = pd.DataFrame(columns=["donor","patient","otu", "lfc","timepoint","aerobic", "donor_abundance"])
    for donor in [131, 485, 500]:
        single_donor_df = all_engraftment[all_engraftment["donor"]==donor]
        for timepoint in unique(metadata.timepoint)[1:-1]: #get rid of nan at the end
            for patient in unique(single_donor_df["patient"].values):
                for otu in unique(single_donor_df["otu"].values):
                    try:
                        pre_index = single_donor_df.index[(single_donor_df["patient"]==patient) & (single_donor_df["otu"]==otu) & (single_donor_df["timepoint"]=="Pre-FMT")][0]
                        post_index = single_donor_df.index[(single_donor_df["patient"]==patient) & (single_donor_df["otu"]==otu) & (single_donor_df["timepoint"]==timepoint)][0]

                        aerobic = single_donor_df.loc[pre_index, "aerobic"]

                        lfc = np.log(single_donor_df.loc[post_index, "abundance"]) - np.log(single_donor_df.loc[pre_index, "abundance"])

                        donor_abundance = single_donor_df.loc[pre_index, "donor_abundance"]
                    except Exception as e:
                        pass

                    lfc_df = lfc_df.append({"donor":donor, "patient":patient, "otu":otu, "lfc":lfc, "timepoint":timepoint, "aerobic":aerobic, "donor_abundance":donor_abundance}, ignore_index=True)
    lfc_df_averaged = lfc_df.drop(["patient"], axis=1).groupby(["otu", "aerobic", "timepoint", "donor"]).mean().reset_index()
    lfc_df_plot = pd.DataFrame(columns=["donor","otu","timepoint","Aerobic", "Anaerobic"])
    for donor in [131, 485, 500]:
        for timepoint in unique(metadata.timepoint)[1:-1]:
            for otu in unique(lfc_df_averaged[lfc_df_averaged["donor"]==donor].otu):
                index = lfc_df_averaged.index[(lfc_df_averaged["donor"]==donor) & (lfc_df_averaged["otu"]==otu) & (lfc_df_averaged["aerobic"]=="Aerobic") & (lfc_df_averaged["timepoint"]==timepoint)][0]
                aerobic = lfc_df_averaged.loc[index, "lfc"]
                anaerobic = lfc_df_averaged.loc[index+1, "lfc"]
                lfc_df_plot = lfc_df_plot.append({"donor":donor,"otu":otu, "timepoint":timepoint, "Aerobic":aerobic, "Anaerobic":anaerobic}, ignore_index=True)
    return lfc_df_plot
lfc_df_plot = lfc_all_timepoints_all_donors(genus_engraftment_direct)
lfc_df_plot_3days = lfc_df_plot[lfc_df_plot["timepoint"]=="3 days"]
lfc_df_plot_8weeks = lfc_df_plot[lfc_df_plot["timepoint"]=="8 weeks"]

print(lfc_df_plot_3days[lfc_df_plot_3days["donor"]==485].head())
plt.close()
sns.set_style("ticks")
sns.lmplot("Aerobic", "Anaerobic", lfc_df_plot_8weeks[lfc_df_plot_8weeks["donor"]==131], fit_reg=False)
plt.xlabel("Log-fold change in abundance (Aerobic)")
plt.ylabel("Log-fold change in abundance (Anaerobic)")
plt.legend(title="")
plt.savefig("lfc_donor131_8weeks.png")
plt.close()

sns.lmplot("Aerobic", "Anaerobic", lfc_df_plot_3days[lfc_df_plot_3days["donor"]==131], fit_reg=False)
plt.xlabel("Log-fold change in abundance (Aerobic)")
plt.ylabel("Log-fold change in abundance (Anaerobic)")
plt.legend(title="")
plt.savefig("lfc_donor131_3days.png")
plt.close()

# sns.lmplot("Aerobic", "Anaerobic", lfc_df_plot_3days[lfc_df_plot_3days["donor"]==500], fit_reg=False)
# plt.xlabel("Log-fold change in abundance (Aerobic)")
# plt.ylabel("Log-fold change in abundance (Anaerobic)")
# plt.legend(title="")
# plt.savefig("lfc_donor500_3days.png")
# plt.close()
#
# sns.lmplot("Aerobic", "Anaerobic", lfc_df_plot_8weeks[lfc_df_plot_8weeks["donor"]==500], fit_reg=False)
# plt.xlabel("Log-fold change in abundance (Aerobic)")
# plt.ylabel("Log-fold change in abundance (Anaerobic)")
# plt.legend(title="")
# plt.savefig("lfc_donor500_8weeks.png")
# plt.close()

#
output_file("lfc_donor131_3days.html")
hover = HoverTool(tooltips=[
    ("Genus", "@otu"), ("Donor", "@donor")])

source = ColumnDataSource(data=lfc_df_plot_3days[lfc_df_plot_3days["donor"]==131])
p = figure(tools=[hover], x_axis_label="Aerobic", y_axis_label="Anaerobic")
# p.circle(x=lfc_df_plot_3days['Aerobic'], y=lfc_df_plot_3days['Anaerobic'], color=["red","green","blue"])
t1 = p.circle(x='Aerobic', y='Anaerobic', color='red',source=source)
# t2 = p.circle(x='Aerobic', y='Anaerobic', color='#4292c6',source=lfc_df_plot_3days[lfc_df_plot_3days["donor"]==485])
# t3 = p.circle(x='Aerobic', y='Anaerobic', color='#9ecae1',source=lfc_df_plot_3days[lfc_df_plot_3days["donor"]==500])
show(p)
output_file("lfc_donor131_8weeks.html")
source = ColumnDataSource(data=lfc_df_plot_8weeks[lfc_df_plot_8weeks["donor"]==131])
p = figure(tools=[hover], x_axis_label="Aerobic", y_axis_label="Anaerobic")
t1 = p.circle(x='Aerobic', y='Anaerobic', color='blue',source=source)
show(p)
output_file("lfc_donor500_3days.html")
source = ColumnDataSource(data=lfc_df_plot_3days[lfc_df_plot_3days["donor"]==500])
p = figure(tools=[hover], x_axis_label="Aerobic", y_axis_label="Anaerobic")
t1 = p.circle(x='Aerobic', y='Anaerobic', color='red',source=source)
show(p)

output_file("lfc_all_donors_8weeks.html")
p = figure(tools=[hover], x_axis_label="Log-fold change in abundance (Aerobic)", y_axis_label="Log-fold change in abundance (Anaerobic)")
t1 = p.circle(x='Aerobic', y='Anaerobic', legend="131",color='blue',source=lfc_df_plot_8weeks[lfc_df_plot_8weeks["donor"]==131])
t2 = p.circle(x='Aerobic', y='Anaerobic', legend="485",color='green',source=lfc_df_plot_8weeks[lfc_df_plot_8weeks["donor"]==485])
t3 = p.circle(x='Aerobic', y='Anaerobic', legend="500",color='red',source=lfc_df_plot_8weeks[lfc_df_plot_8weeks["donor"]==500])
show(p)

output_file("lfc_all_donors_3days.html")
p = figure(tools=[hover], x_axis_label="Log-fold change in abundance (Aerobic)", y_axis_label="Log-fold change in abundance (Anaerobic)")
t1 = p.circle(x='Aerobic', y='Anaerobic', legend="131", color='blue',source=lfc_df_plot_3days[lfc_df_plot_3days["donor"]==131])
t2 = p.circle(x='Aerobic', y='Anaerobic', legend="485", color='green',source=lfc_df_plot_3days[lfc_df_plot_3days["donor"]==485])
t3 = p.circle(x='Aerobic', y='Anaerobic', legend="500", color='red',source=lfc_df_plot_3days[lfc_df_plot_3days["donor"]==500])
show(p)
