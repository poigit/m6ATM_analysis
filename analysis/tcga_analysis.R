### R-4.3.1
# library
library(TCGAbiolinks)
library(SummarizedExperiment)
library(limma)
library(edgeR)
library(biomaRt)
library(glmnet)
library(factoextra)
library(FactoMineR)
library(caret)
library(gplots)
library(RColorBrewer)
library(gProfileR)
library(genefilter)
library(ggpubr)
library(rstatix)
library(pheatmap)

colors = brewer.pal(8, "Set2")

### TCGA data
query = GDCquery(project = 'TCGA-LIHC',
                 data.category = 'Transcriptome Profiling',
                 data.type = 'Gene Expression Quantification',
                 experimental.strategy = 'RNA-Seq',
                 workflow.type = 'STAR - Counts',
                 sample.type = c('Primary Tumor', 'Solid Tissue Normal'))

GDCdownload(query = query)
data = GDCprepare(query = query)



### expression data
d_mat = as.matrix(log2(assay(data, 'fpkm_uq_unstrand')+1))

ensg_full = rownames(d_mat)
ensg_split = strsplit(ensg_full, split = '.', fixed = T)
ensg_split = lapply(ensg_split, `[[`, 1)
rownames(d_mat) = ensg_split

# ENSG to gene symbol
ensembl = useMart('ensembl', dataset = 'hsapiens_gene_ensembl')
ensg_ids = rownames(d_mat)
gene_info = getBM(attributes = c('ensembl_gene_id', 'external_gene_name'),
                  filters = "ensembl_gene_id",
                  values = ensg_ids,
                  mart = ensembl)

colnames(gene_info)[colnames(gene_info) == 'external_gene_name'] = 'name'
gene_info = gene_info[!is.na(gene_info$name) & gene_info$name != '',]
gene_info = gene_info[!(duplicated(gene_info$name)|duplicated(gene_info$name, fromLast = TRUE)),]

d_table = as.data.frame(d_mat)
d_table$ensembl_gene_id = ensg_ids
merged_table = merge(d_table, gene_info, by = 'ensembl_gene_id', all = FALSE)
merged_table = merged_table[!duplicated(merged_table$name),]

rownames(merged_table) = merged_table$name
merged_table = subset(merged_table, select = -c(ensembl_gene_id, name))

# plot heatmap
gene_list = c('PEG10', 'METTL3', 'METTL14', 'METTL16', 'WTAP', 'VIRMA', 'RBM15', 'ZC3H13',
              'FTO', 'ALKBH5', 'YTHDC1', 'YTHDC2', 'YTHDF1', 'YTHDF2', 'YTHDF3', 'EIF3A',
              'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'RBMX', 'HNRNPC', 'HNRNPA2B1')

gene_table = merged_table[rownames(merged_table) %in% gene_list,]

png('FigureS10A.png', units = 'in', width = 18, height = 6, res = 300)

gene_cluster = ifelse(rownames(gene_table) %in% c('PEG10', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3'), '1', '2')
rgroup = data.frame(Cluster = factor(gene_cluster), row.names = rownames(gene_table))
cgroup = data.frame(Group = factor(data$definition), row.names = colnames(gene_table))
breaks = seq(-3, 3, length.out = 101) 
ann_colors = list(Group = c('Primary solid Tumor' = colors[2], 'Solid Tissue Normal' = colors[1]),
                  Cluster = c('1' = colors[4], '2' = colors[5]))
pheatmap(gene_table, scale = 'row', breaks = breaks, annotation_col = cgroup, annotation_colors = ann_colors,
         annotation_row = rgroup, annotation_names_row = F, show_colnames = F, fontsize_row = 15)

dev.off()


### expression analysis
gene_table_t = gene_table[,data$definition == 'Primary solid Tumor']
gene_table_n = gene_table[,data$definition == 'Solid Tissue Normal']

peg10_val = as.numeric(gene_table_t['PEG10',])
q = quantile(peg10_val, c(0.25, 0.75))
q1 = q[1]
q2 = q[2]


peg10_group = ifelse(peg10_val >= q2, 'High',
                     ifelse(peg10_val < q1, 'Low', 'Mid'))

peg10_group = factor(peg10_group, levels = c('Low', 'Mid', 'High'))

# box plot 
png('FigureS10B_IGF2BP1.png', units = 'in', width = 8, height = 6, res = 300)

gene_level = data.frame(level = as.numeric(gene_table_t['IGF2BP1',]), group = peg10_group)
stats = wilcox.test(gene_level[gene_level$group == 'High',]$level,
                    gene_level[gene_level$group == 'Low',]$level,
                    exact = T, paired = T, mu = 0, alternative = 'two.sided')

p = ggplot(gene_level, aes(x = as.factor(group), y = level, color = group, fill = group)) + 
  geom_boxplot(outlier.shape = NA, outlier.size = 2, notch = TRUE, width = 0.5, fill = NA, lwd = 1, fatten = 1) + 
  geom_dotplot(binaxis = 'y',binwidth = 0.3, stackdir = 'center', stackratio = 0, dotsize = 0.5,
               alpha = 0.4) +
  scale_x_discrete(labels = c('PEG10-Low', 'PEG10-Mid', 'PEG10-High')) +
  scale_color_manual(values = c('#1EC1BA', '#2B9ADA', 'orange'))+
  scale_fill_manual(values = c('#1EC1BA', '#2B9ADA', 'orange'))+
  coord_cartesian(ylim = c(-1, 6)) +
  theme_classic() +
  theme(axis.text.x = element_text(size = 28),
        axis.title.x = element_blank(),
        axis.text.y = element_text(size = 28),
        axis.title.y = element_blank(),
        legend.position = 'none')
p

dev.off()

# box plot 
png('FigureS10B_IGF2BP2.png', units = 'in', width = 8, height = 6, res = 300)

gene_level = data.frame(level = as.numeric(gene_table_t['IGF2BP2',]), group = peg10_group)
stats = wilcox.test(gene_level[gene_level$group == 'High',]$level,
                    gene_level[gene_level$group == 'Low',]$level,
                    exact = T, paired = T, mu = 0, alternative = 'two.sided')

p = ggplot(gene_level, aes(x = as.factor(group), y = level, color = group, fill = group)) + 
  geom_boxplot(outlier.shape = NA, outlier.size = 2, notch = TRUE, width = 0.5, fill = NA, lwd = 1, fatten = 1) + 
  geom_dotplot(binaxis = 'y',binwidth = 0.3, stackdir = 'center', stackratio = 0, dotsize = 0.5,
               alpha = 0.4) +
  scale_x_discrete(labels = c('PEG10-Low', 'PEG10-Mid', 'PEG10-High')) +
  scale_color_manual(values = c('#1EC1BA', '#2B9ADA', 'orange'))+
  scale_fill_manual(values = c('#1EC1BA', '#2B9ADA', 'orange'))+
  coord_cartesian(ylim = c(-1, 6)) +
  theme_classic() +
  theme(axis.text.x = element_text(size = 28),
        axis.title.x = element_blank(),
        axis.text.y = element_text(size = 28),
        axis.title.y = element_blank(),
        legend.position = 'none')
p

dev.off()


# box plot 
png('FigureS10B_IGF2BP3.png', units = 'in', width = 8, height = 6, res = 300)

gene_level = data.frame(level = as.numeric(gene_table_t['IGF2BP3',]), group = peg10_group)
stats = wilcox.test(gene_level[gene_level$group == 'High',]$level,
                    gene_level[gene_level$group == 'Low',]$level,
                    exact = T, paired = T, mu = 0, alternative = 'two.sided')

p = ggplot(gene_level, aes(x = as.factor(group), y = level, color = group, fill = group)) + 
  geom_boxplot(outlier.shape = NA, outlier.size = 2, notch = TRUE, width = 0.5, fill = NA, lwd = 1, fatten = 1) + 
  geom_dotplot(binaxis = 'y',binwidth = 0.3, stackdir = 'center', stackratio = 0, dotsize = 0.5,
               alpha = 0.4) +
  scale_x_discrete(labels = c('PEG10-Low', 'PEG10-Mid', 'PEG10-High')) +
  scale_color_manual(values = c('#1EC1BA', '#2B9ADA', 'orange'))+
  scale_fill_manual(values = c('#1EC1BA', '#2B9ADA', 'orange'))+
  coord_cartesian(ylim = c(-1, 6)) +
  theme_classic() +
  theme(axis.text.x = element_text(size = 28),
        axis.title.x = element_blank(),
        axis.text.y = element_text(size = 28),
        axis.title.y = element_blank(),
        legend.position = 'none')
p

dev.off()
