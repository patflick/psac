
# read human 2G data
data2g_fak <- read.csv("../fak_human2g.tsv", sep="\t", header=TRUE)
data2g_dss <- read.csv("../dss_human2g.tsv", header=TRUE)
data2g_our <- read.csv("../psac_human2g.tsv", sep="\t", header=TRUE)

# sequential time
dss_2g_time <- data2g_dss$time

data <- data2g_our[which(data2g_our$p >= 32),]
data$psac <- data$time

fak <- data2g_fak[which(data2g_fak$p >= 32),"time"]
if (length(fak) < length(data$psac)) {
    fak <- c(fak, rep(NA, length(data$psac) - length(fak)))
}

data$fak <- fak

# calculate speedup
data$psac_su <- dss_2g_time / data$psac
data$fak_su <- dss_2g_time / data$fak


# now the same for the full human genome (3G)
data3g_our <- read.csv("../psac_human.tsv", sep="\t", header=TRUE)
data3g_dss <- read.csv("../dss_human.tsv", sep="\t", header=TRUE)

dss_3g_time <- data3g_dss$time

# only 32 <= p <= 1024 processors
data3 <- data3g_our[which(data3g_our$p >= 32),]
data3 <- data3[which(data3$p <= 1024),]

data$psac_3g <- data3$time
data$psac_3g_su <- dss_3g_time / data3$time

prange <- c(1, max(data$p))

data$logp = log2(data$p)
p <- unique(data$p)
logp <- unique(data$logp)


plot(NA,NA,
     # limits
     xlim=range(data$p), ylim=c(0,max(data$psac_3g_su)),
     #xlim=prange, ylim=prange,
     # disable and floor axis
     xaxt="n",
     # titles and labels
     main="Speedup over divsufsort",
     xlab="Number of Cores",
     ylab="Speedup",
     cex.lab=1.3,
     cex.main=1.5
     )


axis(side=1, p, p)

abline(h=1, lty=4)
points(1,1, pch=8)

lines(data$p, data$psac_3g_su, type="o", lty=1, pch=15)
lines(data$p, data$psac_su, type="o", lty=2, pch=17)
lines(data$p, data$fak_su, type="o", lty=3, pch=3)

legend("bottomright", legend=c("Our method", "Our method (2G)", "FAK (CloudSACA) (2G)", "divsufsort (Speedup=1)"),
       lty=c(1,2,3,4), pch=c(15, 17, 3, 8), bg="white")
