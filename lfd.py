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

def genus_engraftment_calculate_lfd(df):
    # """
    # df (DataFrame): columns are "donor", "aerobic", "otu", "abundance" (average of all patients for the same donor and FMT type),
    #     "donor_abundance", "log_abundance" (patient)
    # """
    genus_engraftment_lfd = pd.DataFrame(columns=["donor","otu", "lfd", "only"])
    for donor in df.donor.unique(): #every donor
        for otu in df[df["donor"] == donor].otu: #every otu
            try:
                anaerobic_id = df.index[(df["donor"] == donor) & (df["otu"] == otu) & (df["aerobic"] == "Anaerobic")][0]
                anaerobic = df.loc[anaerobic_id,"log_abundance"]
                donor_anaerobic = df.loc[anaerobic_id,"donor_log_abundance"]
            except:
                anaerobic = -np.inf
            try:
                aerobic_id = df.index[(df["donor"] == donor) & (df["otu"] == otu) & (df["aerobic"] == "Aerobic")][0]
                aerobic = df.loc[aerobic_id,"log_abundance"]
                donor_aerobic = df.loc[aerobic_id,"donor_log_abundance"]
            except:
                aerobic = -np.inf

            if aerobic == -np.inf and anaerobic == -np.inf:
                genus_engraftment_lfd = genus_engraftment_lfd.append({"donor":donor, "otu":otu, "lfd":np.nan, "only":np.nan}, ignore_index=True)
            elif aerobic == -np.inf:
                genus_engraftment_lfd = genus_engraftment_lfd.append({"donor":donor, "otu":otu, "lfd":15, "only":"Anaerobic"}, ignore_index=True)
            elif anaerobic == -np.inf:
                genus_engraftment_lfd = genus_engraftment_lfd.append({"donor":donor, "otu":otu, "lfd":-15, "only":"Aerobic"}, ignore_index=True)
            else:
                lfd = anaerobic - aerobic
                genus_engraftment_lfd = genus_engraftment_lfd.append({"donor":donor, "otu":otu, "lfd":lfd, "only":np.nan}, ignore_index=True)
    return genus_engraftment_lfd
def genus_engraftment_calculate_lfd_pseudocounts(df):
    # """
    # df (DataFrame): columns are "donor", "aerobic", "otu", "abundance" (average of all patients for the same donor and FMT type),
    #     "donor_abundance", "log_abundance" (patient)
    # """
    genus_engraftment_lfd = pd.DataFrame(columns=["donor","otu", "lfd", "donor_lfd"])
    for donor in df.donor.unique(): #every donor
        for otu in df[df["donor"] == donor].otu: #every otu
            try:
                anaerobic_id = df.index[(df["donor"] == donor) & (df["otu"] == otu) & (df["aerobic"] == "Anaerobic")][0]
                anaerobic = df.loc[anaerobic_id,"log_abundance"]
                donor_anaerobic = df.loc[anaerobic_id,"donor_log_abundance"]
            except:
                anaerobic = np.log(0.000001)
                donor_anaerobic = np.log(0.000001)
            try:
                aerobic_id = df.index[(df["donor"] == donor) & (df["otu"] == otu) & (df["aerobic"] == "Aerobic")][0]
                aerobic = df.loc[aerobic_id,"log_abundance"]
                donor_aerobic = df.loc[aerobic_id,"donor_log_abundance"]
            except:
                aerobic = np.log(0.000001)
                donor_aerobic = np.log(0.000001)


            lfd = anaerobic - aerobic
            donor_lfd = donor_anaerobic - donor_aerobic
            genus_engraftment_lfd = genus_engraftment_lfd.append({"donor":donor, "otu":otu, "lfd":lfd, "donor_lfd":donor_lfd}, ignore_index=True)
    return genus_engraftment_lfd

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

genus_engraftment_direct_donor_prefmt = genus_engraftment_direct[(genus_engraftment_direct["timepoint"]=="Pre-FMT") & (genus_engraftment_direct["otu_source"]=="donor")]
genus_engraftment_direct_donor_prefmt_averaged = genus_engraftment_direct_donor_prefmt.drop(['patient', 'timepoint', 'otu_source','donor_direct', 'donor_count'], axis=1)
genus_engraftment_direct_donor_prefmt_averaged["count"] = genus_engraftment_direct_donor_prefmt_averaged['count'].astype(int)
genus_engraftment_direct_donor_prefmt_averaged = genus_engraftment_direct_donor_prefmt_averaged.groupby(['donor', 'aerobic', 'otu']).mean().reset_index()
genus_engraftment_direct_donor_prefmt_averaged["log_abundance"] = np.log(genus_engraftment_direct_donor_prefmt_averaged["abundance"])
genus_engraftment_direct_donor_prefmt_averaged["donor_log_abundance"] = np.log(genus_engraftment_direct_donor_prefmt_averaged["donor_abundance"])

genus_engraftment_direct_donor_prefmt_lfd = genus_engraftment_calculate_lfd_pseudocounts(genus_engraftment_direct_donor_prefmt_averaged)
print(genus_engraftment_direct_donor_prefmt_lfd.head())
#donor composition
sns.set_style("ticks")
sns.barplot(x="otu", y="lfd", hue="donor", palette="RdBu",data=genus_engraftment_direct_donor_prefmt_lfd[genus_engraftment_direct_donor_prefmt_lfd["donor_lfd"]!=0].sort_values(by="donor_lfd"))
plt.xticks(rotation=90, fontsize=40)
plt.yticks(fontsize=40)
plt.figure(figsize=(50,30))
sns.despine(top="True", right="True")
plt.legend(fontsize=40, title="Donor ID")
plt.title("Log-fold difference for genera originating in donors (Donor Composition)", fontsize=60)
plt.ylabel("Log-transformed Ratio of Relative Abundance (Anaerobic:Aerobic)", fontsize=40)
plt.xlabel("Genus", fontsize=40)
plt.savefig("lfd_donor_composition_prefmt.png")
plt.close()
#8 weeks pseudocount
genus_engraftment_direct_donor_8weeks = genus_engraftment_direct[(genus_engraftment_direct["timepoint"]=="8 weeks") & (genus_engraftment_direct["otu_source"]=="donor")]
genus_engraftment_direct_donor_8weeks_averaged = genus_engraftment_direct_donor_8weeks.drop(['patient', 'timepoint', 'otu_source','donor_direct', 'donor_count'], axis=1)
genus_engraftment_direct_donor_8weeks_averaged["count"] = genus_engraftment_direct_donor_8weeks_averaged['count'].astype(int)
genus_engraftment_direct_donor_8weeks_averaged = genus_engraftment_direct_donor_8weeks_averaged.groupby(['donor', 'aerobic', 'otu']).mean().reset_index()
genus_engraftment_direct_donor_8weeks_averaged["log_abundance"] = np.log(genus_engraftment_direct_donor_8weeks_averaged["abundance"])
genus_engraftment_direct_donor_8weeks_averaged["donor_log_abundance"] = np.log(genus_engraftment_direct_donor_8weeks_averaged["donor_abundance"])
genus_engraftment_direct_donor_8weeks_lfd = genus_engraftment_calculate_lfd_pseudocounts(genus_engraftment_direct_donor_8weeks_averaged)
sns.barplot(x="otu", y="lfd", hue="donor", palette="RdBu",data=genus_engraftment_direct_donor_8weeks_lfd[genus_engraftment_direct_donor_8weeks_lfd["lfd"]!=0].sort_values(by="lfd"))
plt.xticks(rotation=90, fontsize=40)
plt.yticks(fontsize=40)
sns.despine(top=True, right=True)
plt.legend(fontsize=40, title="Donor ID")
plt.title("Log-fold difference for genera originating in donors 8 weeks post-FMT", fontsize=60)
plt.ylabel("Log-transformed Ratio of Relative Abundance (Anaerobic:Aerobic)", fontsize=40)
plt.xlabel("Genus", fontsize=40)
plt.figure(figsize=(50,30))
plt.savefig("lfd_patients_8weeks.png")
plt.close()


#3 days pseudocount
genus_engraftment_direct_donor_3days = genus_engraftment_direct[(genus_engraftment_direct["timepoint"]=="3 days") & (genus_engraftment_direct["otu_source"]=="donor")]
genus_engraftment_direct_donor_3days_averaged = genus_engraftment_direct_donor_3days.drop(['patient', 'timepoint', 'otu_source','donor_direct', 'donor_count'], axis=1)
genus_engraftment_direct_donor_3days_averaged["count"] = genus_engraftment_direct_donor_3days_averaged['count'].astype(int)
genus_engraftment_direct_donor_3days_averaged = genus_engraftment_direct_donor_3days_averaged.groupby(['donor', 'aerobic', 'otu']).mean().reset_index()
genus_engraftment_direct_donor_3days_averaged["log_abundance"] = np.log(genus_engraftment_direct_donor_3days_averaged["abundance"])
genus_engraftment_direct_donor_3days_averaged["donor_log_abundance"] = np.log(genus_engraftment_direct_donor_3days_averaged["donor_abundance"])
genus_engraftment_direct_donor_3days_lfd = genus_engraftment_calculate_lfd_pseudocounts(genus_engraftment_direct_donor_3days_averaged)
sns.barplot(x="otu", y="lfd", hue="donor", palette="RdBu", data=genus_engraftment_direct_donor_3days_lfd[genus_engraftment_direct_donor_3days_lfd["lfd"]!=0].sort_values(by="lfd"))
plt.xticks(rotation=90, fontsize=40)
plt.yticks(fontsize=40)
sns.despine(top=True, right=True)
plt.legend(fontsize=40, title="Donor ID")
plt.title("Log-fold difference for genera originating in donors 3 days post-FMT", fontsize=60)
plt.ylabel("Log-transformed Ratio of Relative Abundance (Anaerobic:Aerobic)", fontsize=40)
plt.xlabel("Genus", fontsize=40)
plt.figure(figsize=(50,30))
plt.savefig("lfd_patients_3days.png")
plt.close()
