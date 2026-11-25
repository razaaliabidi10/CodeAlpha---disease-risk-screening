# Streamlit app: disease risk screening
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

from pathlib import Path
PROJECT_DIR = str(Path(__file__).parent)
ACCENT, RISK, GREY, INK = '#1F5C6B', '#B5472F', '#D9D6CE', '#1C1C1C'

st.set_page_config(page_title='Disease Risk Screening', layout='wide')
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
h1, h2, h3 {font-weight: 600; letter-spacing: -0.01em;}
[data-testid="stMetricValue"] {font-size: 2.8rem;}
</style>""", unsafe_allow_html=True)

NAMES = {'heart': 'Heart disease', 'diabetes': 'Diabetes', 'cancer': 'Breast cancer'}
POP = {
    'heart': 'Trained on 303 patients from the Cleveland clinic (UCI), mostly older adults referred for suspected heart problems.',
    'diabetes': 'Trained on 768 women of Pima Indian heritage aged 21+. Not valid for men or other populations.',
    'cancer': 'Trained on 569 fine-needle-aspirate samples (Wisconsin). Inputs are lab measurements of cell nuclei.'
}
OPTIONAL = {'blood_pressure', 'skin_thickness', 'insulin', 'bmi'}
DEFAULT_NA = {'insulin', 'skin_thickness'}
SPECS = {
    'heart': {
        'age': ('int', 'Age (years)'),
        'sex': ('cat', 'Sex', {0: 'Female', 1: 'Male'}),
        'cp': ('cat', 'Chest pain type', {1: 'Typical angina', 2: 'Atypical angina', 3: 'Non-anginal pain', 4: 'Asymptomatic'}),
        'trestbps': ('int', 'Resting blood pressure (mm Hg)'),
        'chol': ('int', 'Serum cholesterol (mg/dl)'),
        'fbs': ('cat', 'Fasting blood sugar above 120 mg/dl', {0: 'No', 1: 'Yes'}),
        'restecg': ('cat', 'Resting ECG', {0: 'Normal', 1: 'ST-T abnormality', 2: 'LV hypertrophy'}),
        'thalach': ('int', 'Maximum heart rate achieved'),
        'exang': ('cat', 'Exercise-induced angina', {0: 'No', 1: 'Yes'}),
        'oldpeak': ('num', 'ST depression (oldpeak)'),
        'slope': ('cat', 'ST slope', {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}),
        'ca': ('cat', 'Major vessels colored by fluoroscopy', {0: '0', 1: '1', 2: '2', 3: '3'}),
        'thal': ('cat', 'Thalassemia', {3: 'Normal', 6: 'Fixed defect', 7: 'Reversible defect'})
    },
    'diabetes': {
        'pregnancies': ('int', 'Pregnancies'),
        'glucose': ('int', 'Plasma glucose (mg/dl)'),
        'blood_pressure': ('int', 'Diastolic blood pressure (mm Hg)'),
        'skin_thickness': ('int', 'Skin fold thickness (mm)'),
        'insulin': ('int', 'Serum insulin (mu U/ml)'),
        'bmi': ('num', 'BMI'),
        'diabetes_pedigree': ('num', 'Diabetes pedigree function'),
        'age': ('int', 'Age (years)')
    }
}


@st.cache_resource
def load_all():
    bundle = joblib.load(f'{PROJECT_DIR}/models/app_bundle.pkl')
    splits = joblib.load(f'{PROJECT_DIR}/data/splits.pkl')
    return bundle, splits


@st.cache_data
def load_tables():
    return {k: pd.read_csv(f'{PROJECT_DIR}/outputs/{k}.csv') for k in ['cv_results', 'final_test_results', 'final_test_all_models']}


BUNDLE, SPLITS = load_all()
T = load_tables()


def spec(name, f):
    return SPECS.get(name, {}).get(f, ('num', f.title()))


def load_patient(name):
    X_te, y_te = SPLITS[name][1], SPLITS[name][3]
    i = np.random.randint(len(X_te))
    row = X_te.iloc[i]
    for f in BUNDLE[name]['features']:
        kind, key, v = spec(name, f)[0], f'{name}_{f}', row[f]
        s = BUNDLE[name]['stats'][f]
        if pd.isna(v):
            if f in OPTIONAL:
                st.session_state[f'{key}_na'] = True
            continue
        if f in OPTIONAL:
            st.session_state[f'{key}_na'] = False
        if kind == 'cat':
            st.session_state[key] = int(v)
        elif kind == 'int':
            st.session_state[key] = int(min(max(v, s['min']), s['max']))
        else:
            st.session_state[key] = float(min(max(v, s['min']), s['max']))
    st.session_state[f'{name}_truth'] = int(y_te.iloc[i])


def reset(name):
    for k in [k for k in st.session_state if k.startswith(f'{name}_')]:
        del st.session_state[k]


def render_feature(name, f, col):
    kind, label, *rest = spec(name, f)
    s = BUNDLE[name]['stats'][f]
    key = f'{name}_{f}'
    if kind == 'cat':
        opts = rest[0]
        keys = list(opts)
        kw = {} if key in st.session_state else {'index': keys.index(min(keys, key=lambda o: abs(o - s['median'])))}
        return float(col.selectbox(label, keys, format_func=opts.get, key=key, **kw))
    na_key = f'{key}_na'
    optional = f in OPTIONAL
    na = st.session_state.get(na_key, f in DEFAULT_NA) if optional else False
    if kind == 'int':
        lo, hi, med, step = int(s['min']), int(s['max']), int(round(s['median'])), 1
    else:
        lo, hi, med = s['min'], s['max'], s['median']
        step = (hi - lo) / 200
    kw = {} if key in st.session_state else {'value': med}
    v = col.slider(label, lo, hi, step=step, key=key, disabled=na, **kw)
    if optional:
        kw2 = {} if na_key in st.session_state else {'value': f in DEFAULT_NA}
        na = col.checkbox('Not measured', key=na_key, **kw2)
        return np.nan if na else float(v)
    return float(v)


def score(name, values):
    pipe = BUNDLE[name]['model']
    X = pd.DataFrame([values], columns=BUNDLE[name]['features']).astype(float)
    p = float(pipe.predict_proba(X)[0, 1])
    prep = pipe.named_steps['prep']
    Xt = prep.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, 'toarray') else np.asarray(Xt)
    contrib = pd.Series(Xt[0] * pipe.named_steps['model'].coef_[0], index=[n.split('__', 1)[-1] for n in prep.get_feature_names_out()])
    return p, contrib


def risk_bar(p, t):
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, x1=1, y0=0, y1=1, fillcolor=GREY, line_width=0)
    fig.add_shape(type='rect', x0=0, x1=p, y0=0, y1=1, fillcolor=RISK if p >= t else ACCENT, line_width=0)
    fig.add_shape(type='line', x0=t, x1=t, y0=-0.25, y1=1.25, line=dict(color=INK, width=2))
    fig.add_annotation(x=t, y=1.5, text=f'threshold {t:.2f}', showarrow=False, font=dict(size=12, color=INK))
    fig.update_xaxes(range=[0, 1], tickformat='.0%', showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, range=[-0.4, 1.9])
    fig.update_layout(height=120, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def contrib_chart(contrib):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).head(8).index).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation='h', marker_color=[RISK if v > 0 else ACCENT for v in top.values]))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis_title='toward healthy  <-    ->  toward sick', showlegend=False)
    fig.update_xaxes(zeroline=True, zerolinecolor=INK, gridcolor=GREY)
    return fig


def tradeoff(name, t):
    y = SPLITS[name][3].values
    p = BUNDLE[name]['model'].predict_proba(SPLITS[name][1])[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    rec = [((p >= g) & (y == 1)).sum() / (y == 1).sum() for g in grid]
    fa = [((p >= g) & (y == 0)).sum() / (y == 0).sum() for g in grid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=rec, name='Sick patients caught', line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=grid, y=fa, name='Healthy patients falsely flagged', line=dict(color=RISK, width=3)))
    fig.add_vline(x=t, line_color=INK, line_width=2)
    fig.update_yaxes(tickformat='.0%', range=[0, 1.02], gridcolor=GREY)
    fig.update_xaxes(title='Flag threshold', gridcolor=GREY)
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h', y=-0.3))
    caught, n_sick = int(((p >= t) & (y == 1)).sum()), int((y == 1).sum())
    flagged, n_healthy = int(((p >= t) & (y == 0)).sum()), int((y == 0).sum())
    return fig, caught, n_sick, flagged, n_healthy


with st.sidebar:
    st.title('Disease Risk Screening')
    name = st.radio('Condition', list(NAMES), format_func=NAMES.get)
    tuned_t = BUNDLE[name]['threshold']
    t = st.slider('Flag threshold', 0.05, 0.95, round(tuned_t, 2), 0.01, key=f'thr_{name}')
    st.caption(f'Tuned threshold: {tuned_t:.2f}. Lower catches more sick patients but raises false alarms.')
    st.divider()
    st.caption('Educational project. Not a diagnostic tool and not medical advice.')

st.title(NAMES[name])
st.caption(POP[name])
tab1, tab2, tab3 = st.tabs(['Predict', 'Model performance', 'About'])

with tab1:
    result = st.container()
    b1, b2, _ = st.columns([1.3, 1, 4])
    b1.button('Load a test patient', on_click=load_patient, args=(name,))
    b2.button('Reset', on_click=reset, args=(name,))
    feats = BUNDLE[name]['features']
    values = {}
    if name == 'cancer':
        groups = {
            'Mean measurements': [f for f in feats if f.startswith('mean')],
            'Error measurements': [f for f in feats if 'error' in f],
            'Worst measurements': [f for f in feats if f.startswith('worst')]
        }
        for gi, (g, fs) in enumerate(groups.items()):
            box = st.container() if gi == 0 else st.expander(g)
            with box:
                if gi == 0:
                    st.markdown(f'**{g}**')
                grid = st.columns(3)
                for i, f in enumerate(fs):
                    values[f] = render_feature(name, f, grid[i % 3])
    else:
        grid = st.columns(3)
        for i, f in enumerate(feats):
            values[f] = render_feature(name, f, grid[i % 3])

    p, contrib = score(name, values)
    flagged = p >= t
    with result:
        c1, c2 = st.columns(2)
        with c1:
            st.metric('Model score', f'{p:.0%}')
            colour = RISK if flagged else ACCENT
            text = 'Flagged: follow-up testing recommended' if flagged else 'Not flagged'
            st.markdown(f"<span style='color:{colour};font-weight:600;font-size:1.15rem'>{text}</span>", unsafe_allow_html=True)
            st.plotly_chart(risk_bar(p, t), width='stretch')
            st.caption('The score is not a calibrated probability. Flagged means worth a clinical follow-up test, not a diagnosis.')
            tk = f'{name}_truth'
            if tk in st.session_state:
                st.caption('Real outcome of the loaded test patient: ' + ('sick' if st.session_state[tk] else 'not sick') + ' (no longer applies once you edit inputs).')
        with c2:
            st.markdown('**What moved this score**')
            st.plotly_chart(contrib_chart(contrib), width='stretch')
            if name == 'cancer':
                st.caption('Cancer measurements are highly correlated, so individual contributions are unstable. Do not read them as causes.')

with tab2:
    st.subheader('Threshold trade-off on held-out test patients')
    fig, caught, n_sick, flagged_n, n_healthy = tradeoff(name, t)
    st.plotly_chart(fig, width='stretch')
    st.markdown(f'At threshold **{t:.2f}**: catches **{caught} of {n_sick}** sick patients and falsely flags **{flagged_n} of {n_healthy}** healthy patients.')
    st.caption('The test set is small, so every number here has a margin of error of several points.')
    st.subheader('Cross-validated comparison of the four algorithms (training data, 5-fold)')
    cv = T['cv_results']
    st.dataframe(cv[cv.dataset == name].drop(columns='dataset'), hide_index=True)
    st.caption('If the gap between two models is smaller than recall_std, the difference is noise.')
    st.subheader('Final test: all four tuned models at the default 0.5 cutoff')
    ta = T['final_test_all_models']
    st.dataframe(ta[ta.dataset == name].drop(columns='dataset'), hide_index=True)
    st.caption('Differences in recall here mostly reflect how each model sets its probabilities, not better separation of patients.')
    st.subheader('Final test: chosen model (Logistic Regression) at both cutoffs')
    tf = T['final_test_results']
    st.dataframe(tf[tf.dataset == name].drop(columns='dataset'), hide_index=True)

with tab3:
    st.markdown("""
**Data.** UCI Heart Disease (Cleveland, 303 patients), Pima Indians Diabetes (768 women), Breast Cancer Wisconsin Diagnostic (569 samples).

**Method.** Missing values are imputed inside a pipeline fitted on training data only. Four algorithms (SVM, Logistic Regression, Random Forest, XGBoost) were compared with 5-fold cross-validation and tuned with grid search. Their scores were statistically indistinguishable, so the simplest and most explainable model, Logistic Regression, was chosen before the test set was opened. The flag threshold was tuned on out-of-fold predictions to reach a recall target.

**Limitations.**
Small datasets, so test results carry a wide margin of error.
Each model reflects its source population only.
Model scores are not calibrated probabilities.
Cancer: 2 of 42 malignant test samples were missed.
Nothing here is clinically validated.
""")
