function varargout = shanghai_dialect_pipeline(action, varargin)
%SHANGHAI_DIALECT_PIPELINE Interpretable Shanghainese binary classifier.
% No input opens the menu. Actions: prepare, train, predict, export.
rootDir = fileparts(mfilename('fullpath'));
if nargin == 0 || isempty(action)
    while true
        fprintf('\n=== 上海话识别实验系统 ===\n1 整理数据  2 训练/评估  3 单文件识别  4 导出 JSON  5 退出\n');
        c = input('请选择: ','s');
        switch c
            case '1', prepare_public_dataset_from_raw();
            case '2', trainModel(rootDir);
            case '3'
                [f,p] = uigetfile({'*.wav;*.mp3;*.m4a;*.flac','Audio'});
                if isequal(f,0), continue; end
                mode=input('阈值模式 balanced/high_precision/high_recall [balanced]: ','s');
                if isempty(mode), mode='balanced'; end
                m=loadModel(rootDir); r=predictFile(m,fullfile(p,f),mode); printResult(r,m);
            case '4', exportJson(loadModel(rootDir),fullfile(rootDir,'shanghai_model.json'));
            case '5', break;
            otherwise, fprintf('无效选项\n');
        end
    end
    return
end
switch lower(char(action))
    case 'prepare', prepare_public_dataset_from_raw();
    case 'train', m=trainModel(rootDir); if nargout, varargout{1}=m; end
    case 'predict'
        if isempty(varargin), error('predict requires an audio path'); end
        mode='balanced'; if numel(varargin)>1, mode=char(varargin{2}); end
        r=predictFile(loadModel(rootDir),char(varargin{1}),mode);
        if nargout, varargout{1}=r; else, disp(r); end
    case 'export'
        p=fullfile(rootDir,'shanghai_model.json'); exportJson(loadModel(rootDir),p);
        if nargout, varargout{1}=p; end
    otherwise, error('Unknown action: %s',char(action));
end
end

function c=cfg(root)
c.root=root; c.data=fullfile(root,'public_dataset','processed','wav16k');
c.index=fullfile(root,'public_dataset','processed','metadata','file_index.csv');
c.modelPath=fullfile(root,'shanghai_model.mat'); c.fs=16000;
c.frame=0.025; c.frameHop=0.010; c.scales=[2 4]; c.windowHop=0.5;
c.minDuration=1; c.minVoicedRatio=0.20; c.minRms=1e-4; c.seed=42;
c.relativeActivityRms=0.08; c.clipAbs=0.995; c.maxClippedRatio=0.02;
c.lowBandHz=[80 400]; c.pitchRangeHz=[80 400]; c.iqrMultiplier=1.5;
c.highPrecisionMinimumRecall=0.50; c.highRecallMinimumPrecision=0.70;
c.split=[.70 .15 .15]; c.maxTrainClassRatio=2; c.minGroups=3;
c.baseNames={'Low-band energy ratio','Pitch standard deviation','Zero-crossing rate', ...
    'Normalized energy variance','Spectral centroid','Spectral flux'};
end

function model=trainModel(root)
c=cfg(root); needToolboxes();
if ~isfile(c.index), error('Run prepare_public_dataset_from_raw() first; file_index.csv is missing.'); end
T=readtable(c.index,'TextType','string');
need={'relative_path','label','source_dataset','speaker_id','file_id'};
if ~all(ismember(need,T.Properties.VariableNames))
    error('file_index.csv missing required columns: %s',strjoin(setdiff(need,T.Properties.VariableNames),', '));
end
X=nan(height(T),12); keep=false(height(T),1);
for i=1:height(T)
    f=fullfile(c.data,char(T.relative_path(i)));
    if ~isfile(f), continue; end
    try
        [X(i,:),q]=featuresForFile(f,c); keep(i)=q.accepted;
    catch ME
        fprintf('[skip] %s: %s\n',char(T.file_id(i)),ME.message);
    end
    if mod(i,500)==0, fprintf('Features %d/%d; usable %d\n',i,height(T),nnz(keep)); end
end
T=T(keep,:); X=X(keep,:);
label=lower(strtrim(string(T.label)));
if any(~ismember(label,["shanghai","non_shanghai"]))
    error('Labels must be shanghai or non_shanghai.');
end
y=double(label=="shanghai");
if numel(unique(y))~=2, error('Both classes are required.'); end
fprintf('Usable clips: %d (Shanghai %d, other %d)\n',numel(y),nnz(y),nnz(~y));
% Group by source and speaker; unknown speakers remain file-specific and are
% flagged in the report rather than presented as verified speaker separation.
g=strings(height(T),1); unknown=false(height(T),1);
for i=1:height(T)
    s=strtrim(string(T.speaker_id(i)));
    if s=="" || lower(s)=="unknown"
        s="file_"+string(T.file_id(i)); unknown(i)=true;
    end
    g(i)=string(T.source_dataset(i))+"|"+s;
end
sp=groupSplit(y,g,c);
tr=sp=="train"; va=sp=="validation"; te=sp=="test";
if any([~any(tr&(y==0)),~any(tr&(y==1)),~any(va&(y==0)),~any(va&(y==1)),~any(te&(y==0)),~any(te&(y==1))])
    error('A speaker split lacks a class. Add speakers or adjust the corpus before training.');
end
% Down-sample only the training majority; validation and test retain prevalence.
rng(c.seed,'twister'); ix0=find(tr&(y==0)); ix1=find(tr&(y==1));
minority=min(numel(ix0),numel(ix1)); cap=c.maxTrainClassRatio*minority;
if numel(ix0)>cap, ix0=ix0(randperm(numel(ix0),cap)); end
if numel(ix1)>cap, ix1=ix1(randperm(numel(ix1),cap)); end
fitIx=[ix0;ix1]; fitIx=fitIx(randperm(numel(fitIx)));
mu=mean(X(fitIx,:),1); sd=std(X(fitIx,:),0,1); sd(~isfinite(sd)|sd<1e-10)=1;
Z=(X-mu)./sd; Xfit=Z(fitIx,:); yfit=y(fitIx);
w=zeros(size(yfit)); w(yfit==0)=numel(yfit)/(2*nnz(yfit==0)); w(yfit==1)=numel(yfit)/(2*nnz(yfit==1));
names={'weighted_threshold','logistic_regression','svm_rbf_auto'};
svmBox=[0.1 1 10]; svmScale=[1 2.5 5];
for bi=1:numel(svmBox)
    for si=1:numel(svmScale)
        names{end+1}=sprintf('svm_rbf_C%.3g_S%.3g',svmBox(bi),svmScale(si)); %#ok<AGROW>
    end
end
names{end+1}='lda';
models=cell(size(names)); scores=cell(size(names)); th=zeros(size(names)); vm=repmat(metrics(),numel(names),1);
for k=1:numel(names)
    models{k}=fitCandidate(names{k},Xfit,yfit,w);
    scores{k}=getScore(models{k},Z(va,:));
    th(k)=f1Threshold(y(va),scores{k});
    vm(k)=metrics(y(va),scores{k}>=th(k),scores{k});
    fprintf('%-22s val F1 %.3f P %.3f R %.3f threshold %.5g\n',names{k},vm(k).f1,vm(k).precision,vm(k).recall,th(k));
end
[~,best]=max([vm.f1]); selected=models{best}; selScore=scores{best};
% Scores are not called probabilities. SVM margins and model scores are
% calibrated monotonically to a bounded validation score for threshold modes.
calibrator=fitclinear(selScore(:),y(va),'Learner','logistic', ...
    'Regularization','ridge','Lambda',0.01,'ClassNames',[0 1]);
valP=calibratedProbability(calibrator,selScore);
thresholds=decisionThresholds(y(va),valP,c);
testScore=getScore(selected,Z(te,:));
testP=calibratedProbability(calibrator,testScore);
testM=metrics(y(te),testP>=thresholds.balanced,testP);
testM.auc=rankAuc(y(te),testP); testM.brier=mean((testP-y(te)).^2);
featureNames=names12(c);
trainPos=tr & (y==1); trainNeg=tr & (y==0);
eff=abs(mean(Z(trainPos,:),1)-mean(Z(trainNeg,:),1));
sources=unique(string(T.source_dataset)); sourceExclusive=0;
for i=1:numel(sources)
    if numel(unique(y(string(T.source_dataset)==sources(i))))==1
        sourceExclusive=sourceExclusive+1;
    end
end
model=selected; model.name=names{best}; model.featureNames=featureNames;
model.mean=mu; model.std=sd; model.thresholds=thresholds;
model.calibrator=calibrator; model.scoreNote='validation-calibrated logistic probability';
model.testMetrics=testM; model.featureEffectSize=eff;
model.trainingDate=datestr(now,30); model.config=c;
model.candidateNames=names; model.validationMetrics=vm; model.validationThresholds=th;
model.testConfusion=confusion(y(te),testP>=thresholds.balanced);
model.dataSummary=struct('usableClips',numel(y),'trainClips',nnz(tr),'valClips',nnz(va), ...
    'testClips',nnz(te),'trainFitClips',numel(fitIx),'speakerGroups',numel(unique(g)), ...
    'unknownSpeakerClips',nnz(unknown),'shanghai',nnz(y==1),'nonShanghai',nnz(y==0), ...
    'sourceCount',numel(sources),'sourceExclusiveCount',sourceExclusive, ...
    'trainShanghai',nnz(tr&(y==1)),'trainOther',nnz(tr&(y==0)), ...
    'valShanghai',nnz(va&(y==1)),'valOther',nnz(va&(y==0)), ...
    'testShanghai',nnz(te&(y==1)),'testOther',nnz(te&(y==0)));
model.manifest=c.index; save(c.modelPath,'model','-v7.3');
writeReport(model,names,vm,c);
showSummary(model,names,vm);
end

function needToolboxes()
for f={'fitclinear','fitcsvm','fitcdiscr'}
    if exist(f{1},'file')~=2, error('Statistics and Machine Learning Toolbox is required (%s missing).',f{1}); end
end
end

function sp=groupSplit(y,g,c)
sp=strings(numel(y),1); rng(c.seed,'twister');
for cls=0:1
    ix=find(y==cls); ug=unique(g(ix));
    if numel(ug)<c.minGroups
        error('Class %d has %d speaker groups; at least %d are required.',cls,numel(ug),c.minGroups);
    end
    n=zeros(numel(ug),1);
    for j=1:numel(ug), n(j)=nnz(g(ix)==ug(j)); end
    order=randperm(numel(ug)); [~,o]=sort(n(order),'descend'); order=order(o);
    target=c.split*numel(ix); counts=zeros(1,3); assignment=zeros(numel(ug),1);
    for j=1:3, assignment(order(j))=j; counts(j)=counts(j)+n(order(j)); end
    for j=4:numel(ug)
        z=order(j); costs=zeros(1,3);
        for p=1:3, trial=counts; trial(p)=trial(p)+n(z); costs(p)=sum(((trial-target)./max(target,1)).^2); end
        [~,p]=min(costs); assignment(z)=p; counts(p)=counts(p)+n(z);
    end
    part=["train","validation","test"];
    for j=1:numel(ug), sp(g==ug(j))=part(assignment(j)); end
end
end

function m=fitCandidate(name,X,y,w)
m=struct('kind',name);
switch name
 case 'weighted_threshold'
    a=mean(X(y==0,:),1); b=mean(X(y==1,:),1); d=abs(b-a);
    m.direction=sign(b-a); m.direction(m.direction==0)=1;
    m.cut=(a+b)/2; m.weights=d/max(sum(d),eps);
 case 'logistic_regression'
    m.classifier=fitclinear(X,y,'Learner','logistic','Regularization','ridge','Weights',w,'ClassNames',[0 1]);
 case 'lda'
    m.classifier=fitcdiscr(X,y,'DiscrimType','linear','Prior',[.5 .5],'Weights',w,'ClassNames',[0 1]);
 otherwise
    if strcmp(name,'svm_rbf_auto')
        box=1; scale='auto';
    else
        tok=regexp(name,'^svm_rbf_C([0-9.]+)_S([0-9.]+)$','tokens','once');
        if isempty(tok), error('Unknown model candidate: %s',name); end
        box=str2double(tok{1}); scale=str2double(tok{2});
    end
    m.kind='svm_rbf'; m.boxConstraint=box; m.kernelScale=scale;
    m.classifier=fitcsvm(X,y,'KernelFunction','rbf','KernelScale',scale, ...
        'BoxConstraint',box,'Standardize',false,'Weights',w,'ClassNames',[0 1]);
end
end

function s=getScore(m,X)
if strcmp(m.kind,'weighted_threshold')
    s=sum(((X-m.cut).*m.direction>0).*m.weights,2); return
end
[~,raw]=predict(m.classifier,X); cn=double(m.classifier.ClassNames(:)');
if size(raw,2)==numel(cn), s=raw(:,find(cn==1,1)); else, s=raw(:,1); end
s=double(s(:));
end

function p=calibratedProbability(calibrator,score)
[~,raw]=predict(calibrator,double(score(:)));
cn=double(calibrator.ClassNames(:)');
p=raw(:,find(cn==1,1)); p=min(max(double(p(:)),0),1);
end

function t=f1Threshold(y,s)
u=unique(s(isfinite(s))); if isempty(u), t=0.5; return; end
v=unique([min(u)-eps(max(abs(u))+1);u;max(u)+eps(max(abs(u))+1)]);
best=-1; t=.5;
for i=1:numel(v), m=metrics(y,s>=v(i),s); if m.f1>best, best=m.f1; t=v(i); end, end
end

function t=decisionThresholds(y,p,c)
t.balanced=f1Threshold(y,p); u=unique([0;p(:);1]); bp=-1; br=-1;
t.high_precision=t.balanced; t.high_recall=t.balanced;
for i=1:numel(u)
    m=metrics(y,p>=u(i),p);
    if m.recall>=c.highPrecisionMinimumRecall && m.precision>bp, bp=m.precision; t.high_precision=u(i); end
    if m.precision>=c.highRecallMinimumPrecision && m.recall>br, br=m.recall; t.high_recall=u(i); end
end
end

function m=metrics(y,pred,s)
if nargin==0
    m=struct('accuracy',0,'precision',0,'recall',0,'f1',0, ...
        'balancedAccuracy',0,'auc',NaN,'brier',NaN);
    return
end
if nargin<3, s=[]; end
y=logical(y(:)); pred=logical(pred(:)); tp=nnz(y&pred); fp=nnz(~y&pred); tn=nnz(~y&~pred); fn=nnz(y&~pred);
m=struct('accuracy',(tp+tn)/max(numel(y),1),'precision',tp/max(tp+fp,1), ...
    'recall',tp/max(tp+fn,1),'f1',0,'balancedAccuracy',.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)),'auc',NaN,'brier',NaN);
m.f1=2*m.precision*m.recall/max(m.precision+m.recall,eps);
if ~isempty(s), m.auc=rankAuc(y,s); end
end

function a=rankAuc(y,s)
y=logical(y(:)); s=double(s(:)); n1=nnz(y); n0=nnz(~y);
if n1==0 || n0==0, a=NaN; return; end
[s,o]=sort(s); y=y(o); r=zeros(size(s)); i=1; base=1;
while i<=numel(s)
    j=i; while j<numel(s)&&s(j+1)==s(i), j=j+1; end
    r(i:j)=mean(base:(base+j-i)); base=base+j-i+1; i=j+1;
end
a=(sum(r(y))-n1*(n1+1)/2)/(n1*n0);
end

function c=confusion(y,p)
y=logical(y(:)); p=logical(p(:)); c=struct('TP',nnz(y&p),'FP',nnz(~y&p),'TN',nnz(~y&~p),'FN',nnz(y&~p));
end

function names=names12(c)
names=cell(1,12); k=0;
for a=1:2, for b=1:6, k=k+1; names{k}=sprintf('%s | %s',sprintf('%gs',c.scales(a)),c.baseNames{b}); end, end
end

function [out,q]=featuresForFile(path,c)
[x,fs]=audioread(path); if isempty(x), error('empty file'); end
x=mean(x,2); x=x-mean(x);
if fs~=c.fs
    if exist('resample','file')==2, x=resample(x,c.fs,fs);
    else, t=(0:numel(x)-1)'/fs; x=interp1(t,x,(0:1/c.fs:t(end))','linear','extrap'); end
end
fs=c.fs;
if numel(x)/fs<c.minDuration, out=nan(1,12); q.accepted=false; return; end
if numel(x)>8*fs, n=round(.05*numel(x)); x=x(n+1:end-n); end
x=x/max(max(abs(x)),1); out=nan(1,12); nValid=0; nTotal=0;
for scale=1:2
    wn=round(c.scales(scale)*fs); hop=round(c.windowHop*fs);
    if numel(x)<=wn, starts=1; else, starts=1:hop:(numel(x)-wn+1); end
    F=[];
    for j=1:numel(starts)
        z=x(starts(j):min(numel(x),starts(j)+wn-1)); nTotal=nTotal+1;
        [f,ok]=windowFeatures(z,fs,c); if ok, F(end+1,:)=f; nValid=nValid+1; end %#ok<AGROW>
    end
    if ~isempty(F), out((scale-1)*6+(1:6))=median(iqrFilter(F,c),1); end
end
q.validRatio=nValid/max(nTotal,1); q.accepted=all(isfinite(out))&&q.validRatio>=c.minVoicedRatio;
end

function [f,ok]=windowFeatures(x,fs,c)
N=round(c.frame*fs); H=round(c.frameHop*fs); nf=1+floor((numel(x)-N)/H);
f=nan(1,6); if nf<3, ok=false; return; end
w=hannWin(N); frames=zeros(N,nf); rms=zeros(nf,1);
for i=1:nf, ix=(i-1)*H+(1:N); v=x(ix); rms(i)=sqrt(mean(v.^2)); frames(:,i)=v.*w; end
active=rms>max(c.minRms,c.relativeActivityRms*max(rms));
if nnz(active)<3||mean(active)<c.minVoicedRatio||mean(abs(x)>c.clipAbs)>c.maxClippedRatio, ok=false; return; end
nfft=2^nextpow2(N); spec=zeros(nfft/2+1,nf); freq=(0:nfft/2)'*fs/nfft;
for i=1:nf, a=abs(fft(frames(:,i),nfft)); spec(:,i)=a(1:nfft/2+1); end
p=spec.^2; band=freq>=c.lowBandHz(1)&freq<=c.lowBandHz(2); low=sum(p(band,active),'all')/(sum(p(:,active),'all')+eps);
cent=sum(freq.*p(:,active),1)./(sum(p(:,active),1)+eps); cent=mean(cent)/fs;
zcr=[]; pitch=[]; energy=[]; flux=[]; prev=[];
for i=find(active)'
    v=frames(:,i); zcr(end+1)=sum(abs(diff(v>=0)))/max(N-1,1); %#ok<AGROW>
    energy(end+1)=rms(i); %#ok<AGROW>
    ph=pitchEstimate(v,fs,c); if isfinite(ph), pitch(end+1)=ph; end %#ok<AGROW>
    cur=spec(:,i)/(sum(spec(:,i))+eps); if ~isempty(prev), flux(end+1)=sum(max(cur-prev,0).^2); end %#ok<AGROW>
    prev=cur;
end
if numel(pitch)<3, pv=0; else, pv=std(pitch); end
energy=energy/(max(energy)+eps); if isempty(flux), fl=0; else, fl=mean(flux); end
f=[low,pv,mean(zcr),var(energy),cent,fl]; ok=all(isfinite(f));
end

function p=pitchEstimate(x,fs,c)
x=x-mean(x); N=numel(x); nfft=2^nextpow2(2*N-1);
a=real(ifft(abs(fft(x,nfft)).^2)); if a(1)<=eps, p=NaN; return; end
a=a/a(1); lo=max(2,floor(fs/c.pitchRangeHz(2))); hi=min(N-2,ceil(fs/c.pitchRangeHz(1)));
if hi<=lo, p=NaN; return; end
[v,k]=max(a(lo+1:hi+1)); if v<.3, p=NaN; else, p=fs/(lo+k-1); end
end

function w=hannWin(n)
if n<=1, w=1; else, w=.5-.5*cos(2*pi*(0:n-1)'/(n-1)); end
end

function F=iqrFilter(F,c)
if size(F,1)<4, return; end
good=true(size(F,1),1);
for j=1:size(F,2), q=quantile(F(:,j),[.25 .75]); d=q(2)-q(1); if d>0, good=good&F(:,j)>=q(1)-c.iqrMultiplier*d&F(:,j)<=q(2)+c.iqrMultiplier*d; end, end
if any(good), F=F(good,:); end
end

function m=loadModel(root)
f=fullfile(root,'shanghai_model.mat'); if ~isfile(f), error('Train the model first.'); end
s=load(f,'model'); m=s.model;
end

function r=predictFile(m,path,mode)
[x,q]=featuresForFile(path,m.config); if ~q.accepted, error('Audio did not pass quality gate.'); end
z=(x-m.mean)./m.std; score=getScore(m,z);
p=calibratedProbability(m.calibrator,score);
if ~isfield(m.thresholds,mode), mode='balanced'; end
r=struct('file',path,'label','non_shanghai','score',score,'decisionScore',p, ...
    'mode',mode,'threshold',m.thresholds.(mode),'features',x,'quality',q);
if p>=r.threshold, r.label='shanghai'; end
end

function printResult(r,m)
fprintf('\nPrediction: %s\nCalibrated score: %.4f | threshold %.4f | mode %s\n',r.label,r.decisionScore,r.threshold,r.mode);
fprintf('Calibration: %s\nModel: %s\n',m.scoreNote,m.name);
figure('Name','Interpretable feature diagnostics','Color','w');
bar(r.features); grid on; xticks(1:numel(m.featureNames)); xticklabels(m.featureNames); xtickangle(35);
ylabel('Feature value'); title('Clip-level features (6 acoustic measures at 2 s and 4 s scales)');
end

function writeReport(m,names,vm,c)
f=fullfile(c.root,'training_report.txt'); id=fopen(f,'w'); if id<0, warning('Could not write report'); return; end
cl=onCleanup(@()fclose(id));
fprintf(id,'Shanghai dialect classification report\nDate: %s\nManifest: %s\n',m.trainingDate,m.manifest);
fprintf(id,'Usable=%d; train=%d; validation=%d; test=%d; unknown-speaker clips=%d\n', ...
    m.dataSummary.usableClips,m.dataSummary.trainClips,m.dataSummary.valClips,m.dataSummary.testClips,m.dataSummary.unknownSpeakerClips);
fprintf(id,'Class counts (Shanghai/other): train %d/%d; validation %d/%d; test %d/%d\n', ...
    m.dataSummary.trainShanghai,m.dataSummary.trainOther,m.dataSummary.valShanghai, ...
    m.dataSummary.valOther,m.dataSummary.testShanghai,m.dataSummary.testOther);
fprintf(id,'Sources=%d; sources containing only one class=%d\n', ...
    m.dataSummary.sourceCount,m.dataSummary.sourceExclusiveCount);
fprintf(id,'Validation model comparison (threshold selected on validation):\n');
for i=1:numel(names), a=vm(i); fprintf(id,'%s accuracy %.4f precision %.4f recall %.4f F1 %.4f\n',names{i},a.accuracy,a.precision,a.recall,a.f1); end
a=m.testMetrics; d=m.testConfusion;
fprintf(id,'Selected: %s\n',m.name);
if strcmp(m.kind,'svm_rbf')
    fprintf(id,'SVM candidate BoxConstraint=%.6g; KernelScale=%.6g\n', ...
        m.boxConstraint,m.classifier.KernelParameters.Scale);
end
fprintf(id,'Test accuracy %.4f balanced_accuracy %.4f precision %.4f recall %.4f F1 %.4f AUC %.4f Brier %.4f\n', ...
    a.accuracy,a.balancedAccuracy,a.precision,a.recall,a.f1,a.auc,a.brier);
fprintf(id,'TP=%d FP=%d TN=%d FN=%d\n',d.TP,d.FP,d.TN,d.FN);
fprintf(id,'Thresholds balanced=%.6g high_precision=%.6g high_recall=%.6g\n', ...
    m.thresholds.balanced,m.thresholds.high_precision,m.thresholds.high_recall);
fprintf(id,'Scores use a ridge-logistic calibration fitted on held-out validation predictions. Feature differences are descriptive, not causal.\n');
fprintf(id,'Threshold policy: balanced=max validation F1; high_precision=max precision with recall >= %.2f; high_recall=max recall with precision >= %.2f.\n', ...
    m.config.highPrecisionMinimumRecall,m.config.highRecallMinimumPrecision);
fprintf(id,'Test performance applies to this corpus and does not guarantee unseen-speaker/domain accuracy.\n');
if min([m.dataSummary.testShanghai,m.dataSummary.testOther])<30
    fprintf(id,'WARNING: fewer than 30 test clips in at least one class; test metrics are highly uncertain.\n');
end
if m.dataSummary.unknownSpeakerClips>0
    fprintf(id,'WARNING: clips without speaker IDs were split by file, not verified speaker; speaker-independent results are not established.\n');
end
if m.dataSummary.sourceExclusiveCount>0
    fprintf(id,'WARNING: at least one recording source contains only one class; the classifier may learn source/channel differences instead of dialect. Collect both classes under matched recording conditions and evaluate on an independent source.\n');
end
end

function showSummary(m,names,vm)
fprintf('\nSelected model: %s\n',m.name);
for i=1:numel(names), a=vm(i); fprintf('%-22s F1 %.3f P %.3f R %.3f\n',names{i},a.f1,a.precision,a.recall); end
a=m.testMetrics; d=m.testConfusion;
fprintf('Untouched test: Acc %.3f Balanced Acc %.3f P %.3f R %.3f F1 %.3f AUC %.3f Brier %.3f\n', ...
    a.accuracy,a.balancedAccuracy,a.precision,a.recall,a.f1,a.auc,a.brier);
fprintf('TP=%d FP=%d TN=%d FN=%d; report: training_report.txt\n',d.TP,d.FP,d.TN,d.FN);
figure('Name','Model diagnostics','Color','w'); tiledlayout(1,2);
nexttile; bar(m.featureEffectSize); grid on; xticks(1:12); xticklabels(m.featureNames); xtickangle(45);
ylabel('|standardized class mean difference|'); title('Descriptive feature separation');
nexttile; imagesc([d.TP d.FN;d.FP d.TN]); axis image; colorbar;
set(gca,'XTick',1:2,'XTickLabel',{'Pred Shanghai','Pred Other'},'YTick',1:2,'YTickLabel',{'Actual Shanghai','Actual Other'});
title(sprintf('Held-out test F1 %.3f',a.f1));
end

function exportJson(m,path)
fc=m.config;
featureConfig=struct('sampleRate',fc.fs,'frameLengthSec',fc.frame, ...
    'frameHopSec',fc.frameHop,'windowScalesSec',fc.scales, ...
    'windowHopSec',fc.windowHop,'minimumDurationSec',fc.minDuration, ...
    'minimumVoicedWindowRatio',fc.minVoicedRatio,'minimumRms',fc.minRms, ...
    'relativeActivityRms',fc.relativeActivityRms,'clippingAbsThreshold',fc.clipAbs, ...
    'maximumClippedRatio',fc.maxClippedRatio,'lowBandHz',fc.lowBandHz, ...
    'pitchRangeHz',fc.pitchRangeHz,'iqrMultiplier',fc.iqrMultiplier, ...
    'highPrecisionMinimumRecall',fc.highPrecisionMinimumRecall, ...
    'highRecallMinimumPrecision',fc.highRecallMinimumPrecision);
p=struct('modelName',m.name,'featureNames',{m.featureNames},'mean',m.mean,'std',m.std, ...
    'classes',{{'non_shanghai','shanghai'}},'positiveClass','shanghai', ...
    'featureConfig',featureConfig,'thresholds',m.thresholds, ...
    'scoreNote',m.scoreNote,'date',m.trainingDate, ...
    'calibrationBeta',m.calibrator.Beta(:)','calibrationBias',m.calibrator.Bias);
if isfield(m,'classifier') && strcmp(m.kind,'svm_rbf')
    z=m.classifier; p.modelType='svm_rbf'; p.supportVectors=z.SupportVectors;
    p.alpha=z.Alpha(:)'; p.bias=z.Bias; p.kernelScale=z.KernelParameters.Scale;
    p.boxConstraint=m.boxConstraint;
elseif isfield(m,'classifier') && strcmp(m.kind,'logistic_regression')
    p.modelType='logistic_regression'; p.beta=m.classifier.Beta(:)'; p.bias=m.classifier.Bias;
elseif strcmp(m.kind,'weighted_threshold')
    p.modelType='weighted_threshold'; p.direction=m.direction; p.cut=m.cut; p.weights=m.weights;
elseif isfield(m,'classifier') && strcmp(m.kind,'lda')
    pair=m.classifier.Coeffs{1,2};
    p.modelType='lda'; p.linear=pair.Linear; p.constant=pair.Const;
else
    error('Selected model type cannot be exported: %s',m.kind);
end
fid=fopen(path,'w'); if fid<0, error('Cannot write %s',path); end
cl=onCleanup(@()fclose(fid)); fwrite(fid,jsonencode(p,'PrettyPrint',true),'char');
fprintf('Exported %s\n',path);
end
