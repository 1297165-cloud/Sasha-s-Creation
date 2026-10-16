import io, re, math, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import sympy as sp
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(page_title='Hispanic Heritage App 2.0', layout='wide')
st.title('Hispanic Heritage App 2.0')
st.caption('Explore country indicators, build functions, and investigate change across disciplines.')
x=sp.Symbol('x', real=True)
CONTEXTS={
 'Business':('Revenue','Time','years','Revenue','USD','120+15*x-0.6*x**2','business'),
 'Engineering':('Projectile height','Time','seconds','Height','meters','2+18*x-4.9*x**2','projectile'),
 'Medical':('Drug concentration','Time','hours','Concentration','mg/L','10*exp(-0.3*x)','tank'),
 'Sports science':('Athlete position','Time','seconds','Position','meters','2*x+0.3*x**2','walker'),
 'Biology':('Population growth','Time','days','Population','cells','100*exp(0.2*x)','population'),
 'Bioengineering':('Tank concentration','Time','minutes','Concentration','mg/L','5+2*sin(x)','tank'),
 'Economic forecasting':('Economic output','Time','years','Output','index','100+3*x-0.08*x**2','business'),
 'Weather':('Temperature','Time','hours','Temperature','°C','18+8*sin(pi*x/12)','thermometer'),
 'Custom':('Quantity','Input','units','Output','units','x**3-3*x','projectile')}

def fetch(country, indicator):
    url=f'https://api.worldbank.org/v2/country/{country}/indicator/{indicator}'
    r=requests.get(url,params={'format':'json','per_page':20000},timeout=20);r.raise_for_status()
    obj=r.json()
    if not isinstance(obj,list) or len(obj)<2 or not isinstance(obj[1],list):raise ValueError('No usable API response.')
    return pd.DataFrame([{'year':int(a['date']),'value':float(a['value'])} for a in obj[1] if a.get('value') is not None and str(a.get('date','')).isdigit()]).sort_values('year').reset_index(drop=True)

def regression(df,degree):
    clean=df[['year','value']].dropna().copy();clean['year']=pd.to_numeric(clean.year,errors='coerce');clean['value']=pd.to_numeric(clean.value,errors='coerce');clean=clean.replace([np.inf,-np.inf],np.nan).dropna()
    if clean.year.duplicated().any():raise ValueError('Duplicate years: keep one value per year.')
    if len(clean)<degree+1:raise ValueError(f'Degree {degree} requires at least {degree+1} distinct years.')
    clean=clean.sort_values('year');base=int(clean.year.min());t=clean.year.to_numpy()-base;y=clean.value.to_numpy()
    with np.errstate(all='ignore'):coeff=np.polyfit(t,y,degree)
    if not np.isfinite(coeff).all():raise ValueError('Unstable fit: reduce polynomial degree.')
    fitted=np.polyval(coeff,t);res=y-fitted;sse=float(np.sum(res**2));sst=float(np.sum((y-y.mean())**2))
    return clean,base,coeff,res, (1-sse/sst if sst>0 else (1. if sse<1e-16 else float('nan'))),float(np.mean(abs(res)))

def parse_expression(source):
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    if len(source)>180 or not re.fullmatch(r'[0-9a-zA-Z_+*/()., ^\-\s]+',source):raise ValueError('Use numbers, x, arithmetic, and supported functions only.')
    names={'x':x,'sin':sp.sin,'cos':sp.cos,'tan':sp.tan,'exp':sp.exp,'log':sp.log,'ln':sp.log,'sqrt':sp.sqrt,'pi':sp.pi,'E':sp.E,'Abs':sp.Abs}
    # Reject unrecognized names before parsing.
    tokens=re.findall(r'[A-Za-z_][A-Za-z_0-9]*',source)
    if any(tok not in names for tok in tokens):raise ValueError('Unknown symbol/function. Allowed: x, sin, cos, tan, exp, log, ln, sqrt, Abs, pi, E.')
    expr=parse_expr(source,local_dict=names,global_dict={'Integer':sp.Integer,'Float':sp.Float,'Rational':sp.Rational,'Pow':sp.Pow,'Mul':sp.Mul,'Add':sp.Add},transformations=standard_transformations+(implicit_multiplication_application,convert_xor),evaluate=True)
    if expr.free_symbols-{x}:raise ValueError('Only x is permitted.')
    return expr

def evaluate(expr,xx):
    fun=sp.lambdify(x,expr,modules=['numpy']);arr=np.asarray(xx,dtype=float)
    try:
        with np.errstate(all='ignore'):v=np.asarray(fun(arr),dtype=float)
        if v.ndim==0:v=np.full(arr.shape,float(v))
        return np.where(np.isfinite(v),v,np.nan)
    except Exception:return np.full(arr.shape,np.nan)

def scalar(expr,z):
    a=evaluate(expr,[z]);return float(a[0])

def extrema_analysis(expr,lo,hi):
    first=sp.diff(expr,x);second=sp.diff(first,x)
    # Symbolic roots for tractable expressions, numerical scanning as a fallback.
    roots=[]
    try:
        sol=sp.solveset(first,x,domain=sp.Interval(lo,hi))
        if isinstance(sol,sp.FiniteSet):
            roots=[float(sp.N(v)) for v in sol if v.is_real and lo<float(sp.N(v))<hi]
    except Exception:pass
    if not roots:
        grid=np.linspace(lo,hi,801);v=evaluate(first,grid)
        for i in range(len(grid)-1):
            if np.isfinite(v[i]) and np.isfinite(v[i+1]) and v[i]*v[i+1]<0:
                try:
                    from scipy.optimize import brentq
                    roots.append(brentq(lambda z:scalar(first,z),grid[i],grid[i+1]))
                except Exception:roots.append((grid[i]+grid[i+1])/2)
    roots=sorted(set(round(v,8) for v in roots))[:60]
    classified=[]
    for r in roots:
        eps=max(1e-5,(hi-lo)*1e-5);left=scalar(first,r-eps);right=scalar(first,r+eps)
        kind='local minimum' if left<0<right else 'local maximum' if left>0>right else 'stationary, not confirmed extremum'
        classified.append((r,scalar(expr,r),kind))
    candidates=[(v,scalar(expr,v)) for v in [lo,hi]+roots if np.isfinite(scalar(expr,v))]
    return first,second,classified,candidates

with st.sidebar:
    st.header('Data & course')
    course=st.selectbox('Course',['Honors Algebra 2','Precalculus','Calculus'],index=2)
    country=st.selectbox('Country',['Mexico','Colombia','Dominican Republic','Puerto Rico','Argentina','Chile','Peru','Brazil','United States'],index=0)
    codes={'Mexico':'MEX','Colombia':'COL','Dominican Republic':'DOM','Puerto Rico':'PRI','Argentina':'ARG','Chile':'CHL','Peru':'PER','Brazil':'BRA','United States':'USA'}
    indicator=st.selectbox('World Bank indicator',['Population, total','GDP (current US$)','Life expectancy at birth (years)','CO2 emissions (metric tons per capita)'])
    indicators={'Population, total':'SP.POP.TOTL','GDP (current US$)':'NY.GDP.MKTP.CD','Life expectancy at birth (years)':'SP.DYN.LE00.IN','CO2 emissions (metric tons per capita)':'EN.ATM.CO2E.PC'}
    degree=st.slider('Polynomial degree',1,6,2)
    st.caption('Data source: World Bank API. Country and indicator coverage varies.')
key=f'{codes[country]}|{indicators[indicator]}'
if st.session_state.get('data_key')!=key:
    st.session_state.data_key=key
    try:st.session_state.original=fetch(codes[country],indicators[indicator]);st.session_state.api_error=''
    except Exception as e:st.session_state.original=pd.DataFrame(columns=['year','value']);st.session_state.api_error=str(e)
    st.session_state.edited=None;st.session_state.results={}
if 'results' not in st.session_state:st.session_state.results={}
st.subheader('1 · Historical data and polynomial fit')
if st.session_state.api_error:st.error('World Bank API unavailable: '+st.session_state.api_error+' — retry by refreshing the page.')
if len(st.session_state.original):
    c1,c2=st.columns([3,1])
    with c1:
        st.caption('Edit year/value cells below. The original data can be restored.')
        source=st.session_state.edited if st.session_state.edited is not None else st.session_state.original
        edited=st.data_editor(source,num_rows='dynamic',hide_index=True,key='editor_'+key)
    with c2:
        if st.button('Apply edits'):
            st.session_state.edited=edited.copy();st.session_state.results={};st.rerun()
        if st.button('Reset to source'):
            st.session_state.edited=None;st.session_state.results={};st.rerun()
    working=st.session_state.edited if st.session_state.edited is not None else st.session_state.original
    try:
        data,base,coeff,res,r2,mae=regression(working,degree)
        fingerprint=(key,degree,tuple((float(a),float(b)) for a,b in data[['year','value']].itertuples(index=False,name=None)))
        if st.session_state.get('fingerprint')!=fingerprint:st.session_state.results={};st.session_state.fingerprint=fingerprint
        polynomial=sum(sp.Float(float(c),15)*(x-base)**(len(coeff)-i-1) for i,c in enumerate(coeff))
        equation=' + '.join(f'({v:.12g})t^{degree-i}' for i,v in enumerate(coeff))
        st.latex(r't=\mathrm{year}-'+str(base)+r',\quad \hat y='+sp.latex(sp.expand(sum(sp.Float(float(c),12)*sp.Symbol('t')**(degree-i) for i,c in enumerate(coeff)))))
        st.caption('Coefficients displayed to 12 significant digits; calculations retain full floating-point precision.')
        m1,m2,m3,m4=st.columns(4)
        m1.metric('In-sample R²',f'{r2:.5f}');m2.metric('In-sample MAE',f'{mae:,.4g}');m3.metric('Observations',len(data));m4.metric('Historical mean',f'{data.value.mean():,.4g}')
        st.write(f'Observed range: {int(data.year.min())}–{int(data.year.max())}; residual = observed − fitted. Units: {indicator}.')
        residuals=data.copy();residuals['fitted']=np.polyval(coeff,data.year-base);residuals['residual']=res
        with st.expander('Observed data, fitted values, residuals'):st.dataframe(residuals,use_container_width=True)
        if len(data)>=max(10,degree+5):
            ntest=max(2,int(len(data)*.2));train=data.iloc[:-ntest];test=data.iloc[-ntest:]
            if len(train)>=degree+1:
                _,b0,c0,_,_,_=regression(train,degree)
                holdout=float(np.mean(abs(test.value-np.polyval(c0,test.year-b0))))
                st.info(f'Chronological holdout MAE: {holdout:,.5g} ({len(train)} earlier training observations, {len(test)} later test observations). This is separate from in-sample metrics.')
        st.subheader('2 · Year calculator')
        a,b=st.columns(2)
        with a:
            yr=st.number_input('Single year',min_value=1950,max_value=2100,value=2025,step=1)
            if st.button('Calculate Value'):
                st.session_state.results['single']=int(yr)
        with b:
            mode=st.radio('Batch input',['Year range','Custom years'],horizontal=True)
            if mode=='Year range':
                start=st.number_input('Start year',1950,2100,2000);end=st.number_input('End year',1950,2100,2030);step=st.slider('Year step',1,10,5)
                requested=list(range(int(start),int(end)+1,step)) if start<=end else []
            else:
                raw=st.text_input('Comma-separated years','2000, 2010, 2025')
                try:requested=[int(s.strip()) for s in raw.split(',') if s.strip()];assert all(1950<=v<=2100 for v in requested)
                except Exception:requested=[];st.warning('Use comma-separated whole years from 1950 through 2100.')
            if st.button('Calculate batch'):
                if not requested:st.error('No valid years: check range order or custom entries.')
                else:st.session_state.results['batch']=requested
        def predict(years):
            vals=np.polyval(coeff,np.array(years)-base)
            return pd.DataFrame({'year':years,'estimate':vals,'classification':['within observed range' if data.year.min()<=z<=data.year.max() else 'outside observed range' for z in years], 'observed actual':[float(data.loc[data.year==z,'value'].iloc[0]) if z in set(data.year) else np.nan for z in years]})
        result_frames=[]
        if 'single' in st.session_state.results:
            single=predict([st.session_state.results['single']]);st.dataframe(single,hide_index=True);result_frames.append(single)
        if 'batch' in st.session_state.results:
            batch=predict(st.session_state.results['batch']);st.dataframe(batch,hide_index=True,use_container_width=True);result_frames.append(batch)
        results=pd.concat(result_frames,ignore_index=True) if result_frames else pd.DataFrame(columns=['year','estimate','classification','observed actual'])
        if len(results) and ((results.estimate<0).any() and (data.value>=0).all()):st.warning('Some polynomial estimates are negative even though historical observations are nonnegative; extrapolation may be implausible.')
        if len(results) and (results.classification=='outside observed range').any():st.warning('Extrapolation beyond observed years is especially uncertain. MAE is an in-sample error summary, not a 95% prediction interval.')
        grid=np.linspace(min(data.year.min(),*(results.year.tolist() or [data.year.min()])),max(data.year.max(),*(results.year.tolist() or [data.year.max()])),450)
        figure=go.Figure();figure.add_scatter(x=data.year,y=data.value,mode='markers',name='Observed')
        inside=grid[(grid>=data.year.min())&(grid<=data.year.max())];outside1=grid[grid<data.year.min()];outside2=grid[grid>data.year.max()]
        figure.add_scatter(x=inside,y=np.polyval(coeff,inside-base),name='Fitted curve')
        for part in [outside1,outside2]:
            if len(part):figure.add_scatter(x=part,y=np.polyval(coeff,part-base),name='Extrapolation',line={'dash':'dash'},showlegend=not any(t.name=='Extrapolation' for t in figure.data))
        for bound in [data.year.min(),data.year.max()]:figure.add_vline(x=bound,line_dash='dot')
        if len(results):figure.add_scatter(x=results.year,y=results.estimate,mode='markers',marker={'symbol':'diamond','size':11},name='Calculated years')
        increment=st.slider('Model sample increment (years)',1,10,5)
        samples=np.arange(int(data.year.min()),int(data.year.max())+1,increment)
        figure.add_scatter(x=samples,y=np.polyval(coeff,samples-base),mode='markers',marker={'size':5},name='Sample markers')
        figure.update_layout(xaxis_title='Year',yaxis_title=indicator,legend_title='Series')
        st.plotly_chart(figure,use_container_width=True)
        st.subheader('3 · Country comparison')
        other=st.selectbox('Compare with', [c for c in codes if c!=country],index=0)
        if st.button('Load comparison'):
            try:
                compare=fetch(codes[other],indicators[indicator]);st.session_state.comparison=(other,compare,key)
            except Exception as e:st.error(f'Comparison fetch failed: {e}')
        if 'comparison' in st.session_state and st.session_state.comparison[2]==key:
            name,otherdata,_=st.session_state.comparison
            comp=go.Figure();comp.add_scatter(x=data.year,y=data.value,name=country);comp.add_scatter(x=otherdata.year,y=otherdata.value,name=name)
            comp.update_layout(xaxis_title='Year',yaxis_title=indicator);st.plotly_chart(comp,use_container_width=True)
            st.caption(f'{country}: {int(data.year.min())}–{int(data.year.max())}; {name}: {int(otherdata.year.min())}–{int(otherdata.year.max())}. Gaps reflect missing source observations; coverage may differ.')
        st.subheader('4 · Exports')
        summary=pd.DataFrame({'field':['country','indicator','base year','degree','in-sample R²','in-sample MAE','observations','historical mean','edited data','equation'], 'value':[country,indicator,base,degree,r2,mae,len(data),data.value.mean(),st.session_state.edited is not None,equation]})
        st.download_button('Download CSV (working data)',data.to_csv(index=False).encode(),file_name='working_data.csv',mime='text/csv')
        excel=io.BytesIO()
        with pd.ExcelWriter(excel,engine='openpyxl') as writer:
            summary.to_excel(writer,sheet_name='Summary',index=False);residuals.to_excel(writer,sheet_name='Working data',index=False);results.to_excel(writer,sheet_name='Calculated years',index=False)
        st.download_button('Download Excel (.xlsx)',excel.getvalue(),file_name='analysis.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        try:
            import plotly.io as pio
            # ReportLab chart drawn directly; no kaleido dependency.
            from reportlab.graphics.shapes import Drawing,PolyLine,String
            drawing=Drawing(480,185);xmin=float(grid.min());xmax=float(grid.max());yv=np.polyval(coeff,grid-base);finite=yv[np.isfinite(yv)];ymin=float(min(data.value.min(),finite.min()));ymax=float(max(data.value.max(),finite.max()));yrange=max(1e-9,ymax-ymin)
            pts=[(35+430*(float(xx)-xmin)/max(1,xmax-xmin),20+145*(float(yy)-ymin)/yrange) for xx,yy in zip(grid,yv) if np.isfinite(yy)]
            drawing.add(PolyLine([v for p in pts for v in p],strokeColor=colors.blue,strokeWidth=1.5))
            drawing.add(String(35,5,'Year',fontSize=8));drawing.add(String(35,172,indicator[:65],fontSize=8))
            pdf=io.BytesIO();doc=SimpleDocTemplate(pdf);styles=getSampleStyleSheet()
            story=[Paragraph('Hispanic Heritage App 2.0 — Model Report',styles['Title'])]
            for line in [f'{country} | {indicator}',f'Historical years: {int(data.year.min())}–{int(data.year.max())} | Observations: {len(data)}',f'Base year: {base} | Polynomial degree: {degree}',f'Equation: {equation}',f'In-sample R²: {r2:.6g} | In-sample MAE: {mae:.6g}',f'Mean: {data.value.mean():.6g} | Edited: {st.session_state.edited is not None}', 'Limitations: polynomial fits describe observed data, not causes. Outside-range estimates may be implausible. MAE is not a prediction interval.']:
                story.extend([Paragraph(str(line).replace('&','&amp;').replace('<','&lt;'),styles['Normal']),Spacer(1,9)])
            story.extend([drawing,Spacer(1,15),Paragraph('Working data preview',styles['Heading2'])])
            preview=[['Year','Observed','Fitted','Residual']]+[[str(int(r.year)),f'{r.value:.5g}',f'{r.fitted:.5g}',f'{r.residual:.5g}'] for r in residuals.head(12).itertuples()]
            story.append(Table(preview,style=[('GRID',(0,0),(-1,-1),.3,colors.grey)]))
            story.append(Paragraph('Calculated results',styles['Heading2']))
            if len(results):story.append(Table([list(results.columns)]+[[str(v) for v in row] for row in results.head(15).itertuples(index=False,name=None)],style=[('GRID',(0,0),(-1,-1),.3,colors.grey)],colWidths=[55,85,160,100]))
            else:story.append(Paragraph('No calculated years yet.',styles['Normal']))
            doc.build(story)
            st.download_button('Download PDF report',pdf.getvalue(),file_name='model_report.pdf',mime='application/pdf')
        except Exception as e:st.warning(f'PDF generation error: {e}')
    except Exception as e:
        st.error(f'Fit or data error: {e}');polynomial=None
else:polynomial=None

st.divider();st.subheader('5 · Context and function laboratory')
context=st.selectbox('Application context',list(CONTEXTS))
default=CONTEXTS[context]
left,right=st.columns(2)
with left:
    description=st.text_input('Context description',default[0]);independent=st.text_input('Independent variable',default[1]);xunits=st.text_input('Independent units',default[2])
with right:
    dependent=st.text_input('Dependent variable',default[3]);yunits=st.text_input('Dependent units',default[4]);visual=st.selectbox('Simulation visual',['walker','projectile','tank','population','thermometer','business'],index=['walker','projectile','tank','population','thermometer','business'].index(default[6]))
choice=st.radio('Function source',['Suggested function','Parameter builder','Custom function','Fitted historical model'],horizontal=True)
family=st.selectbox('Function family',['Polynomial','Rational','Exponential','Logarithmic','Trigonometric'])
if choice=='Suggested function':expression=default[5]
elif choice=='Fitted historical model':
    expression=str(polynomial) if polynomial is not None else default[5]
    st.caption('Historical fit uses calendar year x. Choose an interval near the observed years.')
elif choice=='Custom function':expression=st.text_input('Enter f(x), e.g. x^3-3x, 5/(x-2), 3sin(x)',default[5])
else:
    p,q,r=st.columns(3)
    with p:A=st.number_input('a',value=2.0,step=.5)
    with q:B=st.number_input('b',value=1.0,step=.5)
    with r:C=st.number_input('c',value=0.0,step=.5)
    expression={'Polynomial':f'{A}*x**2+{B}*x+{C}','Rational':f'{A}/(x-({B}))+{C}','Exponential':f'{A}*exp({B}*x)+{C}','Logarithmic':f'{A}*log(x-({B}))+{C}','Trigonometric':f'{A}*sin({B}*x)+{C}'}[family]
lo,hi=st.columns(2)
with lo:lower=st.number_input('Analysis interval start',value=0.0,step=1.0)
with hi:upper=st.number_input('Analysis interval end',value=10.0,step=1.0)
if lower>=upper:st.error('Interval start must be less than interval end.')
else:
    try:
        f=parse_expression(expression);d1=sp.diff(f,x);d2=sp.diff(d1,x)
        st.latex('f(x)='+sp.latex(f))
        if course=='Calculus':st.latex("f'(x)="+sp.latex(d1));st.latex("f''(x)="+sp.latex(d2))
        current=st.slider('Current input',float(lower),float(upper),float((lower+upper)/2),key='current')
        a=st.slider('Secant interval start',float(lower),float(upper),float(lower),key='sec_a')
        b=st.slider('Secant interval end',float(lower),float(upper),float(upper),key='sec_b')
        show_f=st.checkbox('Show function',True);show_d1=st.checkbox('Show first derivative',course=='Calculus');show_d2=st.checkbox('Show second derivative',False)
        show_tangent=st.checkbox('Show tangent line',course=='Calculus');show_secant=st.checkbox('Show secant line',True)
        speed=st.slider('Playback speed (steps per second)',1,10,3)
        buttons=st.columns(3)
        if buttons[0].button('▶ Play'):st.session_state.playing=True
        if buttons[1].button('⏸ Pause'):st.session_state.playing=False
        if buttons[2].button('↺ Reset'):st.session_state.playing=False;st.session_state.current=float(lower);st.rerun()
        if st.session_state.get('playing'):
            st.session_state.current=float(lower if current>=upper else min(upper,current+(upper-lower)/100))
            time.sleep(1/speed);st.rerun()
        xx=np.linspace(lower,upper,1600);fig=go.Figure()
        def add_curve(expr,name,enabled):
            if not enabled:return
            yy=evaluate(expr,xx)
            # Separate nonfinite segments and abrupt jumps near poles.
            threshold=max(1.,np.nanpercentile(abs(yy[np.isfinite(yy)]),75)*8) if np.isfinite(yy).any() else 1.
            yy=np.where(abs(yy)>threshold*100,np.nan,yy)
            jumps=np.abs(np.diff(yy));yy[1:][jumps>threshold]=np.nan
            fig.add_scatter(x=xx,y=yy,name=name,connectgaps=False)
        add_curve(f,'f(x)',show_f);add_curve(d1,"f'(x)",show_d1);add_curve(d2,"f''(x)",show_d2)
        value=scalar(f,current);slope=scalar(d1,current);accel=scalar(d2,current)
        if np.isfinite(value):fig.add_scatter(x=[current],y=[value],mode='markers',marker={'size':13},name='Moving point')
        if show_tangent and np.isfinite(value) and np.isfinite(slope):fig.add_scatter(x=xx,y=value+slope*(xx-current),name='Tangent',line={'dash':'dash'})
        av=float('nan')
        if a!=b and np.isfinite(scalar(f,a)) and np.isfinite(scalar(f,b)):
            av=(scalar(f,b)-scalar(f,a))/(b-a)
            if show_secant:fig.add_scatter(x=xx,y=scalar(f,a)+av*(xx-a),name='Secant',line={'dash':'dot'})
        fig.update_layout(xaxis_title=f'{independent} ({xunits})',yaxis_title=f'{dependent} ({yunits})',height=450)
        st.plotly_chart(fig,use_container_width=True)
        st.write(f'At {independent.lower()} = {current:.4g} {xunits}: {dependent.lower()} = {value:.6g} {yunits}.')
        if course!='Honors Algebra 2':st.write(f'Average rate on [{a:.4g}, {b:.4g}] = {av:.6g} {yunits}/{xunits}.')
        if course=='Calculus':
            st.write(f'Instantaneous rate = {slope:.6g} {yunits}/{xunits}; second derivative = {accel:.6g} {yunits}/{xunits}².')
            st.caption('The first derivative measures instantaneous change; the second derivative measures how that rate changes. For position, these are velocity and acceleration.')
            first,second,critical,candidates=extrema_analysis(f,lower,upper)
            st.write('Critical points found (symbolic where possible; otherwise approximate):',critical or 'None detected')
            if candidates:
                st.write('Absolute extrema among endpoints and detected stationary points (only if continuous on the closed interval):',f'min {min(candidates,key=lambda p:p[1])}; max {max(candidates,key=lambda p:p[1])}')
            st.caption('Critical-point detection is not exhaustive for all transcendental functions. Discontinuities and undefined derivatives require separate checks; endpoint extrema are only guaranteed for continuous functions.')
            # Numerical interval behavior and confirmed concavity transitions.
            test=np.linspace(lower,upper,401);dv=evaluate(d1,test);ddv=evaluate(d2,test)
            st.write('Increasing where f′ > 0; decreasing where f′ < 0. Concave up where f″ > 0; concave down where f″ < 0.')
            inflections=[]
            for j in range(len(test)-1):
                if np.isfinite(ddv[j]) and np.isfinite(ddv[j+1]) and ddv[j]*ddv[j+1]<0:
                    z=(test[j]+test[j+1])/2
                    if np.isfinite(scalar(f,z)):inflections.append(round(float(z),4))
            st.write('Approximate concavity-change candidates (verify continuity):',inflections[:25] or 'None detected')
            st.write('Real domain:',sp.calculus.util.continuous_domain(f,x,sp.S.Reals))
        elif course=='Precalculus':st.caption('Explore degree, coefficients, turning points and concavity through changing average rates; compare secant slopes across intervals.')
        else:st.caption('Explore domain, range, end behavior, trends and average rates of change. Use the graph to estimate the range over the displayed interval.')
        # Meaningful normalized context simulation with explicit scale.
        st.subheader('Context simulation')
        finite=evaluate(f,xx);finite=finite[np.isfinite(finite)]
        ratio=float(np.clip((value-finite.min())/max(1e-12,finite.max()-finite.min()),0,1)) if len(finite) and np.isfinite(value) else 0
        if visual=='walker':st.progress(ratio,text=f'Walking person with prosthetic leg — normalized position {ratio:.0%} along track 🦿')
        elif visual=='projectile':st.progress(ratio,text=f'Moving object — normalized height/position {ratio:.0%} 🚀')
        elif visual=='tank':st.progress(ratio,text=f'Tank / drug level — normalized fill {ratio:.0%} 🧪')
        elif visual=='population':st.progress(ratio,text=f'Population / cell colony — normalized level {ratio:.0%} 🧫')
        elif visual=='thermometer':st.progress(ratio,text=f'Thermometer — normalized temperature {ratio:.0%} 🌡️')
        else:st.progress(ratio,text=f'Business indicator — normalized level {ratio:.0%} 📈')
        st.caption('Simulation is a conceptual indicator scaled to the minimum and maximum finite modeled values on the displayed interval; it is not a physically calibrated animation.')
    except Exception as e:st.error(f'Function/analysis error: {e}')
with st.expander('Assumptions, data flow, and limitations'):
    st.write('World Bank API → editable observations → centered-year polynomial least squares → residuals, predictions, charts and exports. Function lab parses an expression → symbolic derivatives → numerical graph and contextual indicator. Regression is descriptive, not causal; polynomial extrapolation can be unstable. The numerical critical-point scanner may miss roots, poles or oscillations. The domain and continuity of the chosen function must be checked before asserting absolute extrema. In-sample R²/MAE are not out-of-sample accuracy guarantees or confidence intervals. API access requires internet.')
