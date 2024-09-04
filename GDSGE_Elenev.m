% Parameters
parameters APts rho_A AEpisolonVar phi0 phi1 psi eta etaI zetaI Phi g nuB nuS betaB mu_om sig2_A zA deltaK L Lscale alpha gammaS gammaB mu_ZA T bT bTau Pbar1 Pbar2 tauK tauD tauK_int tauPi theta delta mu_G G zeta mu pibar xi BG_guess BG_g sigma_eps rho shieldI kappa chi1adj fracStarget bankDWL outputDWL sigmaI phiI n0 F chiLinear;
% shocks
APts=5;
rho_A=0.4;
AEpisolonVar=0.023^2;

% SDF
phi0=.078;
phi1=0;
g=0.0;

%borrower
betaB=0.94; %discount factor Borrower
nuB=1; %IES 
theta=0.582; %*principal fraction
delta=0.937; %*average life loan pool
F=theta/(1-delta);
psi=2.0;
eta=0.2;

%saver
betaS=0.9815;
nuS=1;

%output
mu_om=1;
sig2_A=0.023^2;
rho_A=0.4;
zA= 0.0; 
mu_ZA=exp(zA+.5*sig2_A/(1-rho_A^2));

% deterministic trend labor-augmenting technological progress
g=0.0;
mu_G=exp(g);
G=mu_G;

%labor
Lscale= 0.7216618597;
gammaS=0.622; %labor income share saver
gammaB=0.378; %labor income share borrower
L = Lscale^gammaB * Lscale^gammaS;
alpha=0.71;


% government
gammaG = 0.172;                 % Government public good expenditures
T = 0.028;                      % *Transfers
bT = -20;                       %* Sensitivity of transfer spending to productivity growth    
bTau = 4.5  ;                      %* Sensitivity of labor taxes to productivity growth
bGamma = -2.0;                   % Sensitivity of government consumption to productivity growth
tauK=0.20;%0.217;                     %* Corporate tax target of 3.41% of GDP
tauPi=tauK;                          %*
tauD=0.132;%0.15;  
%taushield=0.200;                % corp tax rate = tauK / (1 - alpha - taushield);
%alphawedge=0.66/alpha;          % labor share of output = alphawedge * alpha
shieldI = 1;					%* Strength of intermediary deposit tax shield (1 = full shield, 0 no shield)
tau0 = 0.295;% 0.285;            % Labor income tax target of 17.30% of GDP
BG1 = 0.1;
BG2 = 1.2;
exponent=3;
tau_min = 1e-3;
tau_max = 0.6;
BG_min = -0.4;
BG_max = 1.5;
BG_open = 0;
BG_foreign=0;
BG_stv=0.7;
bBG=0.8;

Pbar1=1-0.766;           %* PB
Pbar2=0.766;             %* PS
tauK_int=(1-theta)*tauK;    %*


zeta = 0.6; 
deltaK=0.0825;
xi=0.93;

%intermediary
sigmaI=7.0; 
etaI=0.362;
zetaI=0.332;

%default threshold
pibar=0.004; %calculate the threshold
mu=mu_om;

BG_guess=0.5;
BG_g=BG_guess;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                


sigma_eps=0.019;
rho=0.0;

kappa=0.00084;  
chi1adj=0.14;
chiLinear=0;
fracStarget = 0.12; 
Phi=0.40;
phiI=0.068;


bankDWL=1-0.12;
outputDWL=1;
n0=1.4;

% Shocks
var_shock ZA sigma_om;
shock_num = 10;
[ZATrans,ZA] = markovappr(rho_A,AEpisolonVar^0.5,1,APts);
ZA = exp(ZA(:)');

sigma_om=[0.1, 0.18];
Omprob=[0.91, 0.09; 0.20, 0.80]; %Omega transition matrix

shock_trans = kron(Omprob,ZATrans);

% ENDOGENOUS STATES
% Endogenous variables, bounds, and initial values 
var_state KB LB WI BG;
KB=[1.60,1.75,1.84,1.98,2.05,2.10,2.26,2.40]; %capital
LB=[0.23,0.30,0.33,0.35,0.37,0.39,0.40,0.41,0.42,0.43,0.44,0.46,0.47,0.48,0.49,0.5,0.55]; % LB aggregate leverage of producers
WI=[-0.040,-0.030,-0.020,-0.010,0.000,0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,0.055,0.060,0.065,0.070,0.080,0.090,0.100,0.120,0.140,0.160,0.200,0.270]; % aggregate net worth bank
BG=[0.183,0.467,0.750,1.033,1.317,1.400,1.700]; %Government short term bond

% Endogenous variables, bounds, and initial values
var_policy qB q X cB cS equ eI AI_g AS_g wB wS lamI lamP muI muS VI; 
inbound LamI 0 1;
inbound LamP 0 1;
inbound muI 0 1;
inbound muS 0 1;
inbound qB -10 10;
inbound q -10 10;
inbound X 0 10;
inbound cB 0 10;
inbound cS 0 10;
inbound equ 0 10;
inbound eI 0 10;
inbound AI_g 0 10;
inbound AS_g 0 10;
inbound wB 0 10;
inbound wS 0 10;
inbound VI 0 50;

% Tensor variables
var_tensor netResourcesB tau_eff Y wagebill tauN Tvec1 Tvec2 scriptF ret nstar Y_om_g Y_om sig2 chi1 chi0 z OmA f_om omplus AB OmK Nminus M omminus KB_g AB_g BI_g  F_eps  F_eps_cond F_eps_minus F_eps_plus DI_eff eIcost_intern eIcost_extern cI cI_norm equcost_intern equcost_extern equnorm rD bI bS WS MPK MPL_S MPL_B DebtB BR_expense_all BR_expense1 BR_expense2 PsiS PsiP budgetS Psi DivP dI_eff N AStarget eq_N p PsiSA ZA_factor betaP_g betaB_g betaS_g omstar; 

%var_aux betaP_g betaB_g betaS_g omstar



var_interp cBFuture cSFuture
cBFuture = netResourcesB;
cSFuture = budgetS;
initial cBFuture cB
initial cSFuture cS

% Equations
model;
  eq1 = qB  - lamP*F - GDSGE_EXPECT{betaP_g*(OmA*(1-tauK_int+delta*qB) + f_om*ret/Y_om)};
  eq2 = p - lamP*Phi*p - GDSGE_EXPECT{betaP_g*(p*OmA((1-tauK)*OmK*MPK/mu_om+(1-tauK)*p- (1-tauK)*pibar)- f_om*scriptF*pibar + f_om*omstar*MPK*scriptF/mu_om)};
  eq3 = (1-tauK)*OmA*(Omplus*MPL_B/mu_om-wB)  - f_om*scriptF*(wB-omstar*MPL_B);                                                                                         
  eq4 = (1-tauK)*OmA*(Omplus*MPL_S/mu_om-wS)  - f_om*scriptF*(wS-omstar*MPL_S);
  eq5 = q + shieldI*tauPi*rD - q*lamI - kappa - GDSGE_EXPECT{betaI_g};                                                                                        
  eq6 = qB*(1-xi*lamI) - muI - GDSGE_EXPECT{betaI_g*(M + delta*OmA*qB)};
  eq7 = q + tauD*rD -lamS - GDSGE_EXPECT{betaS_g};
  eq8 = qB + chi1adj*(AS_g/AStarget-1) - GDSGE_EXPECT{betaS_g * (M + delta*OmA*qB)};
  eq9 = (Phi*p*KB_g - F*AB_g)*lamP;  
  eq10 = (xi*qB*AI_g + q*bI)*lamI;
  eq11 = AI_g*muI; 
  eq12 = AS_g*muS;
  eq13 = BG - bS - bI; 
  eq14 = AB_g - AI_g - AS_g; 
  eq15 = cB - (1-tau_eff)*wB*Lscale+ p*X - X - Psi*KB + DivP + dI_eff + Tvec1 + BR_expense1;
  eq16 = VI - phiI*WI + eI - GDSGE_EXPECT{betaB_g*(F_eps_minus + (1-F_eps)*rho)/(1-betaB_g*F_eps)}; %2.13


  %SDF
  equnorm = 1/(1-phi1*equ);
  betaB_g = betaB *(cB/cBFuture)*exp(-g / nuB); %borrower A.4
  betaP_g = betaB_g * (phi0/equnorm + 1-phi0); %producer A.14
  betaS_g = betaS *(cS/cSFuture)*exp(-g / nuS); %saver A.38
  

  %net resource
  BR_expense_all = (1-eta)*(zeta*(1-OmA)*(1-deltaK)*p*KB + zeta*outputDWL*(1-OmK)*ZA*KB^(1-alpha)*L^alpha) + bankDWL*(1-etaI)*(1-F_eps)*zetaI*DebtB + PsiS + PsiP + pibar*KB;
  BR_expense1 = Pbar1 * BR_expense_all;
  BR_expense2 = Pbar2 * BR_expense_all;
  netResourcesB = (1-tau_eff)*wB*Lscale + Tvec1 + BR_expense1 + p*X + dI_eff + DivP;
  DebtB=AI_g*(M + qB*(delta*OmA))/G;
  tau_eff=tauN * ZA_factor^bTau; % procyclical
  tauN=BG/Y;
  Tvec1=Pbar1 * T * Y * ZA_factor^bT; % countercyclical
  Tvec2=Pbar2 * T * Y * ZA_factor^bT; % countercyclical
  DivP = N*(phi0 - eq_N ) - (1-OmA)*n0;
  dI_eff = phiI*WI - eI  - F_eps_minus - (1-F_eps)*WI;

  %BR_expense = Pbar*(1-eta)*BR_loss;
  %BR_loss = zeta*(1-OmA)*(1-deltaK)*p*KB_g/G + zeta*params.outputDWL*(1-OmK)*mu_ZA*(KB_g/G)^(1-alpha)*L^alpha + etaI/eta*bankDWL*zetaI*(1-F_eps)*DebtB + PsiS + pibar*KB_g + PsiP + sigmaI/2*eI^2;

  %output
  ZA_factor=ZA/mu_ZA;
  mu_ZA=exp(zA+.5*sig2_A/(1-rho_A^2));

  %wage
  wagebill=Lscale*wB + Lscale*wS;

  %aggregate dividend to borrowers
  N = (1-zeta)*(1-tauPi)*omplus*Y_om - OmA*(1-tauPi)*wagebill + OmA*((1-zeta)*(1-(1-tauPi)*deltaK)*p - (1-tauPi)*pibar)*KB - OmA*(1-tauPi*(1-theta)+delta*qB)*AB  + (1-OmA)*n0;
  equcost_intern = phi1/2 * eq_N^2;
  eq_N = equ/N; %equ is eq eq_N is eq/N ep/np

  %corporate default
  scriptF = nstar/Y_om_g;
  ret = (1-tauPi)*mu_om*ZA*KB^(1-alpha)*L^alpha- (1-tauPi)*wagebill+ (p*(1-(1-tauPi)*deltaK) - (1-tauPi)*pibar)*KB- AB*(1-tauPi+delta*qB); %A.9
  nstar = ret/N;
  Y_om_g = Y_om/N;

  %default threshold
  Y=mu_om*ZA*KB^(1-alpha)*L^alpha;
  Y_om=Y/mu_om;
  omstar=(AB + wagebill + pibar*KB)/Y_om;
  omstar=real(omstar);

  % borrower payoff risk
  AB=LB*p*KB/qB;   %outstanding debt LB leverage KB capital

  %payoff_gamma
  sig2=sig_om^2;
  chi1 = sig2./mu;
  chi0 = mu./chi1;
  z = omstar ./ chi1;
  %OmA = 1 - gamcdf(omstar, chi0, chi1);
  f_om = omstar.^(chi0-1) .* exp(-z) ./ ( gamma(chi0) .* chi1.^chi0);
  %omplus = mu.*(1-gamcdf(omstar, chi0+1, chi1));
  OmK = omplus;

  % payoff to intermediary
  M= OmA*(1-tauPi_int) + Nminus/(AB_g/G); %2.11
  omminus=(1-OmK);
  Nminus= (1-zeta)*(1-tauK)*omminus*Y_om-(1-OmA)*(1-tauK)*wagebill+(1-OmA)*((1-zeta)*(1-(1-tauK)*deltaK)*p-(1-tauK)*pibar)*KB;

  % producer 
  %X=(G-1 + deltaK)*KB_g/G;
  KB_g=(1 - deltaK + X/KB)*KB;
  AB_g = (p*KB_g - (1-phi0)*N - equ + equcost_intern)/qB; %  AB_g bond hold by firms

  % capital price
  p=1 + psi*(X/KB-(mu_G-1+deltaK));  

  % intermediary
  BI_g=-xi*qB*AI_g/q;   

  BS_g=BG_g-BI_g;                                                                                         

  betaI_g=betaB_g*F_eps*(phiI/cI_norm + (1-phiI)); %A.29

  DI_eff=phiI*WI - eI - eIcost_extern - F_eps_minus-(1-F_eps)*WI;

  eIcost_intern = sigmaI/2*eI^2;
  eIcost_extern = 0;
  cI = 1 - sigmaI*eI;
  cI_norm = 1/cI;

  equcost_extern = 0;
  equnorm = 1/(1-phi1*equ);

  % compute intermediary default rate
  %F_eps_next=fastnormcdf(VI_next+rho,0,sigma_eps);
  %f_eps_next=normpdf(VI_next+rho,0,sigma_eps);
  %F_eps_minus_next = -sigma_eps*normpdf((VI_next+rho)/sigma_eps);

  F_eps=fastnormcdf(VI+rho,0,sigma_eps);
  F_eps_cond=normpdf((VI+rho)/sigma_eps);
  F_eps_minus=-sigma_eps*F_eps_cond;
  F_eps_plus=sigma_eps*F_eps_cond;
  %sigmaI=sigmaI*F_eps^sigIexp;

  % intermediary intermediary
  rD=1/q-1;
  bI=(WI - phiI*WI + eI - eIcost_intern - qB*AI_g)/(q+shieldI*tauPi*rD-kappa); %constraint condition

  % saver
  bS=( (1-tau_eff)*wS*Lscale + WS + Tvec2 + BR_expense2 - cS - qB*AS_g - PsiS)/(q+tauD*rD);
  budgetS=(1-tau_eff)*wS*Lscale + WS + Tvec2 + BR_expense2 - qB*AS_g - PsiS;
  WS=(AS_g*(M + qB*(delta*OmA))+BS_g)/G;
  %WI=(AI_g*(M + qB*(delta*OmA))+BI_g)/G; %wealth of bank

  PsiP = N*phi1/2*eq_N^2;
  PsiS = PsiSA + chiLinear * AS_g;
  PsiSA = chi1adj/2 * (AS_g / AStarget - 1)^2 * AStarget; 


  % probability of default for intermediaries
  %AS_g = fracS * AB_g;
  AStarget = fracStarget * AB_g;
  AI_g=AB_g - AS_g;
  % fraction corporate debt held by savers
  %fracS = AS_g ./ AB_g;

  MPK=(1-alpha)*mu_om*mu_ZA*((KB_g/G)/L)^(-alpha);
  MPL_S= alpha*gammaS*mu_om*mu_ZA*L/Lscale*((KB_g/G)/L)^(1-alpha);
  MPL_B= alpha*gammaB*mu_om*mu_ZA*L/Lscale*((KB_g/G)/L)^(1-alpha);

  % investment and capital
  Psi=psi/2*(X/KB-(mu_G-1+deltaK))^2;

  % multiplier transformations
  %muRplus=max(0,muR)^3;
  %muRminus=max(0,-muR)^3;
  %lamRplus=max(0,lamR)^3;
  %lamRminus=max(0,-lamR)^3;
  %lamSplus=max(0,lamS)^3;
  %lamSminus=max(0,-lamS)^3;
  %muSplus=max(0,muS)^3;
  %muSminus=max(0,-muS)^3;
  %lamPplus=max(0,lamP)^3;
  %lamPminus=max(0,-lamP)^3;


  equations;
    eq1;
    eq2;
    eq3;
    eq4;
    eq5;
    eq6;
    eq7;
    eq8;
    eq9;
    eq10;
    eq11;
    eq12;
    eq13;
    eq14;
    eq15;
    eq16;
  end;
end;

simulate;
  num_periods = 10000;
  num_samples = 100;
  initial X 0.5;
  initial shock 1;
  var_simu qB q equ eI AI_g AS_g;

end;













