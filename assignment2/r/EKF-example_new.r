##----------------------------------------------------------------
## EKF algorithm for use in Part 4 of computer exercise 2 in
## Advanced Time Series Analysis
##----------------------------------------------------------------

##----------------------------------------------------------------
## Do the simulation here and keep the values in y
##----------------------------------------------------------------


##----------------------------------------------------------------
## Estimation with the EKF
##----------------------------------------------------------------
## aInit : The starting guess of the AR coefficient estimate
aInit <- 0.5
## aVarInit : The initial variance for estimation of the AR coefficient
aVarInit <- 1
## sigma.v : Standard deviation of the system noise of x in the filter
sigma.v <- 1

## Initialize
## Init the state vector estimate
zt <- c(0,aInit)
## Init the variance matrices
Rv <- matrix(c(sigma.v^2,0,0,0), ncol=2)
## sigma.e : Standard deviation of the measurement noise in the filter
Re <- 1 

## Init the P matrix, that is the estimate of the state variance
Pt <- matrix(c(Re,0,0,aVarInit), nrow=2, ncol=2)
## The state is [X a] so the differentiated observation function is
Ht <- t(c(1,0))
## Init a vector for keeping the parameter a variance estimates
aVar <- rep(NA,length(y))
## and keeping the states
Z <- matrix(NA, nrow=length(y), ncol=2)
Z[,1] <- zt

## The Kalman filtering
for(t in 1:(length(y)-1))
  {
  ## Derivatives (Jacobians)
  Ft <- matrix(c(zt[2],0,zt[1],1), ncol=2)  # F_t-1
  # Ht does not change 
  
  ## Prediction step
  zt = c(zt[2]*zt[1],zt[2]) #z_t|t-1 f(z_t-1|t-1)
  Pt = Ft %*% Pt %*% t(Ft) + Rv #P_t|t-1
  
  ## Update step
  res = y[t] - zt[1] # the residual at time t
  St =  Ht %*% Pt %*% t(Ht) + Re # innovation covariance
  Kt = Pt %*% t(Ht) %*% St^-1 # Kalman gain
  zt = zt + Kt * res # z_t|t
  Pt = (diag(2) - Kt%*%Ht)%*%Pt #P_t|t
  
  ## Keep the state estimate
  Z[t+1,] <- zt
  ## Keep the P[2,2], which is the variance of the estimate of a
  aVar[t+1] <- Pt[2,2]
  
  }
