require(ggplot2)
require(plyr)

#data <- read.csv("./benchmark_local_8.csv", sep=";", header=TRUE)
data <- read.csv("./cyence_methods.csv", sep=";", header=TRUE)

seq_data <- read.csv("./divsufsort.csv", header=FALSE)
colnames(seq_data) <- c("time")

seq_time = median(seq_data$time)/1000

cdata <- ddply(data, c("p", "method"), summarise, time=median(time)/1000)
cdata_noslow <- cdata[which(cdata$method != "reg-lcp" & cdata$method != "reg-nolcp"),]

#w <- 800
#h <- 600
w <- 7
h <- 4

# runtime plot
fig <- ggplot(cdata, aes(x=p, y=time, group=method, lty=method, color=method)) +
        geom_line() +
        scale_x_continuous(breaks = unique(cdata$p),trans="log2") +
        expand_limits(y=0) +
        xlab("Number of cores") +
        ylab("Time [s]") +
        labs(title="Runtime of our methods (Human Genome)")
pdf("runtime_methods.pdf", width=w, height=h)
plot(fig)
dev.off()

# runtime plot (excluding the slow algorithms)
fig_noslow <- ggplot(cdata_noslow, aes(x=p, y=time, group=method, lty=method, color=method)) +
        geom_line() +
        scale_x_continuous(breaks = unique(cdata$p),trans="log2") +
        expand_limits(y=0) +
        xlab("Number of cores") +
        ylab("Time [s]") +
        labs(title="Runtime of our methods (Human Genome)")
pdf("runtime_methods_noslow.pdf", width=w, height=h)
plot(fig_noslow)
dev.off()

# Speedup plot (vs libdivsufsort)
fig_su <- ggplot(cdata, aes(x=p, y=seq_time/time, group=method, lty=method, color=method)) +
        geom_line() +
        scale_x_continuous(breaks = unique(cdata$p),trans="log2") +
        expand_limits(y=0) +
        xlab("Number of cores") +
        ylab("Speedup") +
        labs(title="Speedup over libdivsufsort (Human Genome)")
pdf("speedup_divsufsort.pdf", width=w, height=h)
plot(fig_su)
dev.off()
