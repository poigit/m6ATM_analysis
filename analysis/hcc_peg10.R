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

advanced = meta[meta$class == 'ad',]$analysis_name
early = meta[meta$class == 'e',]$analysis_name
normal = meta[meta$class == 'n',]$analysis_name

gene_table = d_table[rownames(d_table) %in% c('PEG10'),]
gene_table_a = gene_table[,advanced]
gene_table_e = gene_table[,early]
gene_table_n = gene_table[,normal]


gene_table_ordered = cbind(gene_table_a, gene_table_e, gene_table_n)
gene_level = data.frame(level = as.numeric(gene_table_ordered['PEG10',]),
                        group = c(rep('Advanced', dim(gene_table_a)[2]),
                                  rep('Early', dim(gene_table_e)[2]),
                                  rep('Normal', dim(gene_table_n)[2])))


png('FigureS8.png', units = 'in', width = 8, height = 6, res = 300)

p = ggplot(gene_level, aes(x = as.factor(group), y = level, color = group, fill = group, alpha = 0.5)) + 
  geom_boxplot(width = 0.5, lwd = 1) +
  scale_color_manual(values = c('#1EC1BA', '#2B9ADA', 'orange')) +
  scale_fill_manual(values = c('#1EC1BA', '#2B9ADA', 'orange'))+
  ylab('PEG10 level (log2)')+
  theme_classic() +
  theme(axis.text.x = element_text(size = 20),
        axis.title.x = element_blank(),
        axis.text.y = element_text(size = 20),
        axis.title.y = element_text(size = 32),
        legend.position = 'none')
p

dev.off()

stats = wilcox.test(gene_level[gene_level$group == 'Advanced',]$level,
                    gene_level[gene_level$group == 'Normal',]$level,
                    exact = T, paired = F, mu = 0, alternative = 'two.sided')
