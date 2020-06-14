#!/usr/bin/python

''' Credits:
    -------
    Author: Rahul Raj (@rahulrajpl)
    License: MIT License 2020
    

    Reference:
    ----------
    [1] https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/notebook
    [2] https://coolsymbol.com/emojis

'''

import streamlit as st
import pandas as pd
import numpy as np
import time
import seaborn as sns
import pandas_profiling
import webbrowser
from matplotlib import pyplot as plt


def main():
    
    st.title('📊WExDA')
    st.subheader('Web based tool for Exploratory Data Analysis' )
   
    
    @st.cache(persist=True)
    def load_data(uploaded_file):
        df = pd.read_csv(uploaded_file)
        return df


    uploaded_file = st.file_uploader('Upload CSV file to begin (Max file size allowed: 200MB)', type='csv')
    
    if uploaded_file is not None:
        df = load_data(uploaded_file) 
        
        st.sidebar.title('Tools 🔧')  
        if st.checkbox('Show raw data', value=False):
            st.write(df)

        target_column = st.selectbox('Select Target Column', list(df.columns), key='target_column')
        if target_column is not None:
            if st.sidebar.checkbox('One Click Pandas Profiling Report', key='one_click_report_btn'):
                profiling_option = st.sidebar.selectbox('Select method of profiling', ('--select--','Minimal', 'Full Profiling(very slow)'))
                if profiling_option == 'Minimal':
                    with st.spinner('Running pandas_profiling... (it usually take atlest 5 minutes). To cancel, refresh the page'):
                        pandas_profiling.ProfileReport(df, minimal=True).to_file("report.html")
                        url = "./report.html"
                        st.balloons()
                        st.success('Pandas Profiling Completed')
                        time.sleep(2)
                    webbrowser.open(url,new=2)
                elif profiling_option == 'Full Profiling(very slow)':
                    with st.spinner('Running pandas_profiling... (it usually take atlest 5 minutes). To cancel, refresh the page'):
                        pandas_profiling.ProfileReport(df).to_file("report.html")
                        url = "./report.html"
                        st.balloons()
                        st.success('Pandas Profiling Completed')
                        time.sleep(2)
                    webbrowser.open(url,new=2)
                else:
                    st.info('Select the method for profiling')
            if st.sidebar.checkbox('Describe ✍', value=False):
                st.markdown('## Data Description')
                st.write(df.describe())
                st.markdown('### Columns that are potential binary features')
                bin_cols = []
                for col in df.columns:
                    if len(df[col].value_counts()) == 2:
                        bin_cols.append(col)    
                st.write(bin_cols)
                st.markdown('### Columns Types')  
                st.write(df.dtypes)

            if st.sidebar.checkbox('Missing Data 👁', value=False):
                st.markdown('## Missing Data')
                total = df.isnull().sum().sort_values(ascending=False)
                percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
                missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
                st.write(missing_data)

            if st.sidebar.checkbox('Value Counts 🔢', value=False):
                st.markdown('## Value Counts')
                col = st.selectbox('Select Column', list(df.columns), key='val_col')
                st.write(df[col].value_counts())

            if st.sidebar.checkbox('Unique elements 🔗', value=False):
                st.markdown('## Unique elements')
                if st.checkbox('Show all unique elements', value=False):
                    st.write(df.nunique())
                col = st.selectbox('Show columnwise unique elements',list(df.columns),key='unique_col')
                st.write(df[col].unique())

            if st.sidebar.checkbox('Show Distribution 〽', False):
                st.subheader(f'Distribution of {target_column}')
                try:
                    sns.distplot(df[target_column])
                    st.write("Skewness: %.3f" % df[target_column].skew())
                    st.write("Kurtosis: %.3f" % df[target_column].kurt())
                    st.pyplot()
                except:
                    st.error('Invalid Column')

            if st.sidebar.checkbox('Scatter Plot 📈', value=False):
                scatter_cols = st.sidebar.multiselect('Select Column', list(df.columns), key='scatter_cols')
                st.markdown('## Scatter Plots')
                for col in scatter_cols:
                    try:
                        data = pd.concat([df[target_column], df[col]], axis=1)
                        data.plot.scatter(x=col, y=target_column, ylim=(0,800000))
                        st.pyplot()
                    except:
                        st.error('Invalid column')

            if st.sidebar.checkbox('Box Plot 🈁', value=False):
                box_cols = st.sidebar.multiselect('Select Column', list(df.columns), key='box_cols')
                st.markdown('## Box Plots')
                for col in box_cols:
                    try:
                        data = pd.concat([df[target_column], df[col]], axis=1)
                        f, ax = plt.subplots(figsize=(8, 6))
                        fig = sns.boxplot(x=col, y=target_column, data=data)
                        fig.axis(ymin=np.min(df[target_column]), ymax=np.max(df[target_column]))
                        st.pyplot()
                    except:
                        st.error('Invalid column')
            
            if st.sidebar.checkbox('Pair Plot ➿', value=False):
                pair_cols = st.sidebar.multiselect('Select Column', list(df.columns), key='pair_plot')
                plot_size = st.sidebar.number_input('Select Plot size', 1.0, 5.0, step=0.5, key='plot_size', value=2.5)
                st.markdown('## Pair Plots')
                cols = [target_column]
                for col in pair_cols:
                    cols.append(col)
                try:
                    sns.set()
                    sns.pairplot(df[cols], size = plot_size)
                    st.pyplot()
                except:
                    st.error('Invalid column')

            if st.sidebar.checkbox('Correlation matrix 🧮', value=False):
                st.markdown('## Correlation matrix (heatmap style)')
                corrmat = df.corr()
                f, ax = plt.subplots(figsize=(12, 9))
                sns.heatmap(corrmat, vmax=.8, square=True)
                st.pyplot()

                if st.checkbox('With Target Column', value=False):
                    k = st.number_input('# of Cols for heatmap', 3, len(df.columns), step=1, key='k') #number of variables for heatmap
                    cols = corrmat.nlargest(k, target_column)[target_column].index
                    cm = np.corrcoef(df[cols].values.T)
                    sns.set(font_scale=1.25)
                    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
                    st.pyplot()

            if st.sidebar.button('Credits'):
                st.sidebar.markdown('''
                🙋‍♂️

                MIT License 2020 (c) **Rahul Raj**

                Get in touch: [Twitter](https://twitter.com/@rahulrajpl)

                Source Code: [Github](https://github.com/rahulrajpl/wexda)
                ''')


if __name__ == '__main__':
    main()