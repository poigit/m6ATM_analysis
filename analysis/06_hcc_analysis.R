### R-4.3.1
# library
library(edgeR)
library(RColorBrewer)
library(pheatmap)

colors = brewer.pal(8, "Set2")

### HCC data
data = read.table('../data/pivot_160_FRepr.txt', sep = '\t', header = T)
rownames(data) = data$Gene
data = data[-c(1:13)]

group = factor(ifelse(grepl('T', colnames(data)), 'Tumor', 'Normal'))
group = relevel(group, ref = 'Normal')
design = model.matrix(~group)

dge = DGEList(counts = data, genes = as.data.frame(rownames(data)))

# filtering
keep = filterByExpr(dge, design, min.count = 1, min.total.count = 1)
dge = dge[keep,,keep.lib.sizes = FALSE]
rm(keep)

### expression data
d_mat = as.matrix(log2(dge$counts+1))
d_table = as.data.frame(d_mat)

###
meta = read.table('../data/pivot_meta.csv', sep = ',', header = T)

group_cols = replace(meta$class, meta$class=='ad', 'advanced')
group_cols = replace(group_cols, group_cols=='e', 'early')
group_cols = replace(group_cols, group_cols=='n', 'normal')

# plot heatmap
gene_list = c('PEG10', 'METTL3', 'METTL14', 'METTL16', 'WTAP', 'VIRMA', 'RBM15', 'ZC3H13',
              'FTO', 'ALKBH5', 'YTHDC1', 'YTHDC2', 'YTHDF1', 'YTHDF2', 'YTHDF3', 'EIF3A',
              'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'RBMX', 'HNRNPC', 'HNRNPA2B1')

gene_table = d_table[rownames(d_table) %in% gene_list,]

png('Figure6B.png', units = 'in', width = 18, height = 6, res = 300)

gene_cluster = ifelse(rownames(gene_table) %in% c('PEG10', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3'), '1', '2')
rgroup = data.frame(Cluster = factor(gene_cluster),row.names = rownames(gene_table))
cgroup = data.frame(Group = factor(group_cols), row.names = colnames(gene_table))
breaks = seq(-2, 2, length.out = 101) 
ann_colors = list(Group = c('advanced' = colors[2], 'early' = colors[3], 'normal' = colors[1]),
                  Cluster = c('1' = colors[4], '2' = colors[5]))

pheatmap(gene_table, scale = 'row', breaks = breaks, annotation_col = cgroup, annotation_colors = ann_colors,
         annotation_row = rgroup, annotation_names_row = F, show_colnames = F, fontsize_row = 15)

dev.off()


### expression analysis
gene_table_t = gene_table[,group == 'Tumor']
gene_table_n = gene_table[,group == 'Normal']

peg10_val = as.numeric(gene_table_t['PEG10',])
q = quantile(peg10_val, c(0.25, 0.75))
q1 = q[1]
q2 = q[2]


peg10_group = ifelse(peg10_val >= q2, 'High',
                      ifelse(peg10_val < q1, 'Low', 'Mid'))

peg10_group = factor(peg10_group, levels = c('Low', 'Mid', 'High'))

# box plot 
png('Figure6C_IGF2BP1.png', units = 'in', width = 8, height = 6, res = 300)

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
png('Figure6C_IGF2BP2.png', units = 'in', width = 8, height = 6, res = 300)

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
png('Figure6C_IGF2BP3.png', units = 'in', width = 8, height = 6, res = 300)

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
