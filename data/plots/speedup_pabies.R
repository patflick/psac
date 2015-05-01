

# load data
data_human <- read.csv("../psac_human.tsv", sep="\t", header=TRUE)
data_human_lcp <- read.csv("../psac_lcp_human.tsv", sep="\t", header=TRUE)
data_pabies <- read.csv("../psac_pabies.tsv", sep="\t", header=TRUE)
data_pabies_lcp <- read.csv("../psac_lcp_pabies.tsv", sep="\t", header=TRUE)

# filter Human by p >= 128
data_human <- data_human[which(data_human$p >= 128 & data_human$p != 1280),]
data_human_lcp <- data_human_lcp[which(data_human_lcp$p >= 128 & data_human_lcp$p != 1280),]
data_pabies <- data_pabies[which(data_pabies$p != 1280),]
data_pabies_lcp <- data_pabies_lcp[which(data_pabies_lcp$p != 1280),]

# assign runtime
data <- data_pabies
data$pabies <- data$time
data$human <- data_human$time
data$pabies_lcp <- data_pabies_lcp$time
data$human_lcp <- data_human_lcp$time


# calculate self speedup
data$human_su <- data$human[1] / data$human
data$human_lcp_su <- data$human_lcp[1] / data$human_lcp
data$pabies_su <- data$pabies[1] / data$pabies
data$pabies_lcp_su <- data$pabies_lcp[1] / data$pabies_lcp

prange <- c(1, max(data$p))

data$logp = log2(data$p)
p <- unique(data$p)
logp <- unique(data$logp)


plot(NA,NA,
     # limits
     xlim=range(data$p), ylim=c(0,max(10)),
     #xlim=prange, ylim=prange,
     # disable and floor axis
     xaxt="n",
     # titles and labels
     main="Self-Speedup",
     xlab="Number of Cores",
     ylab="Speedup",
     cex.lab=1.3,
     cex.main=1.5
     )

axis(side=1, p, p)


#abline(h=1, lty=4)
#points(1,1, pch=8)

lines(data$p, data$human_su, type="o", lty=1, pch=15)
lines(data$p, data$human_lcp_su, type="o", lty=3, pch=4)
lines(data$p, data$pabies_su, type="o", lty=1, pch=17)
lines(data$p, data$pabies_lcp_su, type="o", lty=3, pch=3)


legend("bottomright", legend=c("Human Genome (SA)", "Human Genome (SA + LCP)", "Pine Genome (SA)", "Pine Genome (SA + LCP)"),
       lty=c(1,3,1,3), pch=c(15, 4, 17, 3), bg="white")
