#
#  Plotting different sections (including LCP construction) for human genome
#

data <- read.csv("../sections.tsv", sep="\t", header=TRUE)


data$logp = log2(data$p)

#ordered_data <- unique(data[order(data$p),])
#p <- unique(ordered_data$p)

#logp <- log2(p)

# time in seconds!
data$time <- data$time / 1000

plot(NA,NA,
     # limits
     xlim=range(data$logp), ylim=c(0,max(data$time)),
     # disable and floor axis
     yaxs="i", xaxs="i", xaxt="n",
     # titles and labels
     main="Runtime composition",
     xlab="Number of Processors",
     ylab="Runtime [s]",
     cex.lab=1.3,
     cex.main=1.5
     )


ticks=c(1, 2, 3, 4, 5, 7)
p <- unique(data$p)
logp <- unique(data$logp)
axis(side=1, logp[ticks], p[ticks])

# don't plot total time (since it is implicit)
data <- data[which(data$section != "total"),]

# choose legend (fill, angle etc)
dens <- c(NA, NA, NA, NA, 15)
colors <- c("grey10","grey40", "grey60", "grey90", NA)
angles <- c(NA, NA, NA, NA, 45)


# get sections
sections <- unique(data$section)
secdata <- data[which(data$section == "LCP"),]

i <- 1
secdata <- data[which(data$section == "LCP"),]
secdata <- secdata[order(secdata$logp),]
prev_p <- secdata$logp
prev_y <- rep(0, length(secdata$time))
for (sec in sections)
{

secdata <- data[which(data$section == sec),]
secdata <- secdata[order(secdata$logp),]

secdata$time <- secdata$time + prev_y

polygon(c(secdata$logp, rev(prev_p)), c(secdata$time, rev(prev_y)), density=dens[i], col=colors[i], angle=angles[i])

prev_p <- secdata$logp
prev_y <- secdata$time

i <- i + 1
}
colors[5] <- "black"
dens[5] <- 30
sections <- data.frame(lapply(sections, as.character), stringsAsFactors=FALSE)
sections[5] <- "LCP Construction"
sections[3] <- "Algorithm 1 (remaining)"
sections[1] <- "sort tuples (samplesort)"
legend("topright", legend=rev(sections), fill=rev(colors), density=rev(dens) )#, angle=rev(angles))
