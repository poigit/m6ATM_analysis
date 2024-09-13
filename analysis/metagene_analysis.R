### R-4.3.1
# library 
library('Guitar')
library(RColorBrewer)

# colors 
colors = brewer.pal(8, "Set2")

### Figure3D HEK293
# Bed
stBedFiles = list('../data/m6atm_hek293.bed', '../data/gt_sac_hek293.bed', '../data/gt_cims_hek293.bed')

# Guitar Plot
png('Figure3D.png', units = 'in', width = 12, height = 6, res = 300)
p = GuitarPlot(txGenomeVer = 'hg38',
               stBedFiles = stBedFiles,
               headOrtail = TRUE,
               enableCI = FALSE,
               mapFilterTranscript = TRUE,
               pltTxType = c('mrna'),
               stGroupName = c('m6ATM', 'm6A-SAC-seq', 'miCLIP-seq'))

theme_set(theme_classic())
p = p + scale_color_manual(values = colors[c(2,1,3)]) + scale_fill_manual(values = c('white', colors[1], 'white'))
p = p + theme(text = element_text(size = 28), legend.text=element_text(size = 28))
p

dev.off()


### Figure5C HepG2
# Bed
stBedFiles = list('../data/m6atm_hepg2.bed', '../data/gt_sac_hepg2.bed')

# Guitar Plot
png('Figure5C.png', units = 'in', width = 12, height = 6, res = 300)
p = GuitarPlot(txGenomeVer = 'hg38',
               stBedFiles = stBedFiles,
               headOrtail = TRUE,
               enableCI = FALSE,
               mapFilterTranscript = TRUE,
               pltTxType = c('mrna'),
               stGroupName = c('m6ATM', 'm6A-SAC-seq'))

theme_set(theme_classic())
p = p + scale_color_manual(values = colors[c(2,1)]) + scale_fill_manual(values = c('white', colors[1]))
p = p + theme(text = element_text(size = 28), legend.text=element_text(size = 28))
p

dev.off()
