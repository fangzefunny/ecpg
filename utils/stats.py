import pingouin as pg 

def t_test(x_data, y_data, paired=False, title=''):
    df = pg.ttest(x_data, y_data, paired=paired)
    dof = df.loc[:, 'dof'].values[0]
    t   = df.loc[:, 'T'].values[0]
    pval = df.loc[:, 'p-val'].values[0]
    cohen_d = df.loc[:, 'cohen-d'].values[0]
    ci = df.loc[:, 'CI95%'].values[0]
    pair_str = '-paired' if paired==True else ''
    print(f'{title} \tt{pair_str}({dof:.3f})={t:.3f}, p={pval:.3f}, cohen-d={cohen_d:.3f}, 95% CI={ci}')

def corr(x_data, y_data, title=''):
    df = pg.corr(x_data, y_data)
    n = df.loc[:, 'n'].values[0]
    r   = df.loc[:, 'r'].values[0]
    pval = df.loc[:, 'p-val'].values[0]
    ci = df.loc[:, 'CI95%'].values[0]
    print(f'{title} \tr({n})={r:.3f}, p={pval:.3f}, 95% CI={ci}')

def anova(dv, between, data, all_table=False):
    df = pg.anova(dv=dv, between=between, data=data).rename(columns={'p-unc': 'punc'})
    if all_table:
       print(df.round(3).to_string())
    else:
        sig_df = df.query('punc<=.05')
        dof2 = int(df.query('Source=="Residual"')['DF'].values[0])
        other_min_p = df.query('punc>.05')['punc'].min()
        for _, row in sig_df.iterrows():
            title = row['Source']
            dof   = int(row['DF'])
            F     = row['F']
            p     = row['punc']
            np2   = row['np2']
            print(f'\t{title}:\tF({dof}, {dof2})={F:.3f}, p={p:.3f}, np2={np2:.3f}')
        print(f'\tOther: \tp>={other_min_p:.3f}')

def linear_regression(x, y, add_intercept=False, title='', x_var='x', y_var='y'):
    df = pg.linear_regression(X=x, y=y, add_intercept=add_intercept)
    beta0 = df['coef'][0]
    beta1 = df['coef'][1]
    pval  = df['pval'][1]
    print(f'{title}\t{y_var}={beta1:.3f}{x_var}+{beta0:.3f},\n\tp={pval:.3f}')
