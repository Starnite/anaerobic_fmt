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
from scipy.stats import mannwhitneyu
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

otu = pd.read_csv("all_filtered_otu_table.csv", index_col=0).transpose()
metadata= pd.read_csv("sample_metadata.csv", index_col=0)
seq_freq_abundance = pd.read_csv("all_filtered_otu_table.csv", index_col=0).transpose().drop(["ndc432", "ndc442"])
seq_freq = pd.read_csv("all_filtered_otu_table_counts.csv", index_col=0).transpose().drop(["ndc432", "ndc442"])
# for every patient, filter out seq vars with only one count (important for analysis of # seq vars in each patient)
seq_freq_filtered_by_patient = seq_freq.copy()
seq_freq_filtered_by_patient=seq_freq_filtered_by_patient.replace(1,0)

seq_var_dict = {} #dict where keys are patient ids and values are lists of indices representing seq vars that are in
                    #the corresponding donor but not in the patient originally, where the first list is direct and the second pma
for i in seq_freq.index:
    if metadata.loc[i, "timepoint"] == "Pre-FMT":
        seq_var_dict[metadata.loc[i, "person_id"]] = [list(set(list(seq_freq.loc[metadata.loc[i, "direct_donor_fmt_id"]].nonzero()[0]))-set(list(seq_freq.loc[i].nonzero()[0]))),list(set(list(seq_freq.loc[metadata.loc[i, "pma_donor_fmt_id"]].nonzero()[0]))-set(list(seq_freq.loc[i].nonzero()[0])))]

donor1=[] #500 - An:143, A:146
donor2=[] #485 - An:141, 159, A:176
donor3=[] #131 - An: 140, 167, A: 154, 160, 163, 178

engraftment_summary_df = pd.DataFrame(columns=["timepoint", "direct/pma", "count", "freq","abundance", "aerobic"])
with PdfPages("engraftment_timeseries.pdf") as pp:
    for patient, donor_seqs in seq_var_dict.items():
        donor_seqs.append(set(donor_seqs[0])-set(donor_seqs[1]))
        engraftment_df = pd.DataFrame(columns=["timepoint", "direct/pma", "count", "freq","abundance"])
        for i, timepoint in enumerate(["3 days", "10 days", "4 weeks", "8 weeks"]):
            try:

                #sample_id of sample corresponding to the given patient and timepoint
                sample_id = metadata.index[(metadata["person_id"] == patient) & (metadata["timepoint"] == timepoint)][0]
                #list of indices of seq vars from direct and pma-processed donor samples that also show up in patient at different timepoints

#                 direct = list(set(donor_seqs[0]) & set(list(seq_freq.loc[sample_id].nonzero()[0])))
#                 pma = list(set(donor_seqs[1]) & set(list(seq_freq.loc[sample_id].nonzero()[0])))
#                 direct_pma = list(set(donor_seqs[2]) & set(list(seq_freq.loc[sample_id].nonzero()[0])))

                direct = list(set(donor_seqs[0]) & set(seq_freq.loc[sample_id]))
                pma = list(set(donor_seqs[1]) & set(seq_freq.loc[sample_id]))
                direct_pma = list(set(donor_seqs[2]) & set(seq_freq.loc[sample_id]))

                direct_abundance = 0
                pma_abundance = 0
                direct_pma_abundance = 0

                for d in direct:
                    direct_abundance += seq_freq_abundance.loc[sample_id,seq_freq_abundance.columns[d]]
                for p in pma:
                    pma_abundance += seq_freq_abundance.loc[sample_id,seq_freq.columns[p]]
                for dp in direct_pma:
                    direct_pma_abundance += seq_freq_abundance.loc[sample_id,seq_freq.columns[dp]]
                aerobic = metadata.loc[sample_id, "anaerobic_fmt"]
                engraftment_df = engraftment_df.append({"timepoint":timepoint, "direct/pma":"direct", "count":len(direct), "freq":len(direct)/len(seq_freq.loc[sample_id]), "abundance":direct_abundance}, ignore_index=True)
                engraftment_df = engraftment_df.append({"timepoint":timepoint, "direct/pma":"pma", "count":len(pma), "freq":len(pma)/len(seq_freq.loc[sample_id]), "abundance":pma_abundance}, ignore_index=True)
                engraftment_df = engraftment_df.append({"timepoint":timepoint, "direct/pma":"direct-pma", "count":len(direct_pma), "freq":len(direct_pma)/len(seq_freq.loc[sample_id]), "abundance":direct_pma_abundance}, ignore_index=True)
                engraftment_summary_df = engraftment_summary_df.append({"timepoint":timepoint, "direct/pma":"direct", "count":len(direct), "freq":len(direct)/len(seq_freq.loc[sample_id]), "abundance":direct_abundance, "aerobic":aerobic}, ignore_index=True)
                engraftment_summary_df = engraftment_summary_df.append({"timepoint":timepoint, "direct/pma":"pma", "count":len(pma), "freq":len(pma)/len(seq_freq.loc[sample_id]), "abundance":pma_abundance,"aerobic":aerobic}, ignore_index=True)


            #missing timepoints
            except Exception as e:
                print(e)
        if metadata.loc[sample_id, "donor_fmt_num"] == 500:
            donor1.append([engraftment_df, int(patient), metadata.loc[sample_id,"anaerobic_fmt"]])
        elif metadata.loc[sample_id, "donor_fmt_num"] == 485:
            donor2.append([engraftment_df, int(patient), metadata.loc[sample_id,"anaerobic_fmt"]])
        else:
            donor3.append([engraftment_df, int(patient), metadata.loc[sample_id,"anaerobic_fmt"]])
        ax1 = sns.barplot(x="timepoint", y="count", hue="direct/pma",data = engraftment_df)
        ax1.set_title("Patient: {} Donor: {} {}".format(int(patient), int(metadata.loc[sample_id,"donor_fmt_num"]), metadata.loc[sample_id,"anaerobic_fmt"]))
        ax1.set_ylabel("total # donor seq vars in patient")
        pp.savefig()
        plt.close()

        ax2 = sns.barplot(x="timepoint", y="abundance", hue="direct/pma",data = engraftment_df)
        ax2.set_title("Patient: {} Donor: {} {}".format(int(patient), int(metadata.loc[sample_id,"donor_fmt_num"]), metadata.loc[sample_id,"anaerobic_fmt"]))
        ax2.set_ylabel("total relative abundance of donor seq vars in patient")
        pp.savefig()
        plt.close()

data = engraftment_summary_df[engraftment_summary_df['direct/pma']=="direct"]
aerobic = data[(data["aerobic"]=="Aerobic") & (data["timepoint"]=="10 days")]["abundance"]
anaerobic = data[(data["aerobic"]=="Anaerobic") & (data["timepoint"]=="10 days")]["abundance"]
print(aerobic, anaerobic)
mwu, pval = mannwhitneyu(aerobic, anaerobic)
print(mwu, pval)
