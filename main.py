import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy
import os
from datetime import datetime, timedelta
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk

url = "https://covid19.who.int/WHO-COVID-19-global-data.csv"
file_path = os.path.join("data", "covid")
os.makedirs(file_path, exist_ok=True)
csv_path = os.path.join(file_path, "WHO-COVID-19-global-data.csv")
df = pd.read_csv(csv_path)
df_index = df.index
df_cols = df.columns
today = datetime.today() - timedelta(days=1)
today = today.strftime('%Y-%m-%d')


def total_deaths(country):
    country = df.loc[(df.Country == country), ['Date_reported', 'Cumulative_deaths']]
    country = country.loc[(country.Date_reported == today), ["Cumulative_deaths"]]
    if len(country.Cumulative_deaths.values) < 1:
        return 0
    td = country.Cumulative_deaths.values[0]
    return td


def total_deaths_world_func():
    world = df.loc[(df.Date_reported == today), ["Cumulative_deaths"]]
    world = world.Cumulative_deaths.sum()
    return world


def total_cases(country):
    country = df.loc[(df.Country == country), ['Date_reported', 'Cumulative_cases']]
    country = country.loc[(country.Date_reported == today), ["Cumulative_cases"]]
    if len(country.Cumulative_cases.values) < 1:
        return 0

    country = country.Cumulative_cases.values[0]
    return country


def total_cases_world_func():
    world = df.loc[(df.Date_reported == today), ["Cumulative_cases"]]
    world = world.Cumulative_cases.sum()
    return world


def avg_deaths_pd(country):
    country = df.loc[
        (df.Country == country) & ((df['Date_reported'] > '2020-01-03 ') & (df['Date_reported'] <= today)) &
        (df.Cumulative_cases > 0), ['Date_reported', 'Country_code', 'Country', 'WHO_region', 'New_cases',
                                    'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']]
    x = country.loc[country.Date_reported == today]
    x = x.Cumulative_deaths.values[0]
    return int(x / len(country))


def avg_deaths_pd_world():
    world = df.loc[(df.Date_reported == today), ['Date_reported', 'Cumulative_deaths']]
    world = world.Cumulative_deaths.sum()
    len(df["Country"].unique())
    x = len((df[(df['Date_reported'] > '2020-01-03 ') & (df['Date_reported'] <= today) & (df.Cumulative_cases > 0)]))
    y = len(df["Country"].unique())
    total_days = x / y
    return int(world / total_days)


def avg_new_case_pd(country):
    country = df.loc[(df.Country == country) & ((df['Date_reported'] > '2020-01-03 ') & (df['Date_reported'] <= today))
                     & (df.Cumulative_cases > 0), ['Date_reported', 'Country_code', 'Country', 'WHO_region',
                                                   'New_cases', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']]
    x = country.loc[country.Date_reported == today]
    x = x.Cumulative_cases.values[0]
    x = x / len(country)
    return int(x)


def avg_new_case_pd_world():
    world = df.loc[(df.Date_reported == today), ['Date_reported', 'Cumulative_cases']]
    world = world.Cumulative_cases.sum()
    len(df["Country"].unique())
    x = len((df[(df['Date_reported'] > '2020-01-03 ') & (df['Date_reported'] <= today) & (df.Cumulative_cases > 0)]))
    y = len(df["Country"].unique())
    total_days = x / y
    return int(world / total_days)


def highest_death_day(country):
    x = df.loc[(df.Country == country), ['Date_reported', 'Country_code', 'Country', 'WHO_region', 'New_cases',
                                         'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']]
    return x.New_deaths.max()


def highest_death_day_world():
    death_day = pd.DataFrame(columns=['date', "deaths"])
    for d in df['Date_reported'].unique():
        x = df.loc[(df.Date_reported == d), ['New_deaths', 'Date_reported']]
        x = x.New_deaths.sum()
        death_day = death_day.append({'date': d, 'deaths': x}, ignore_index=True)
    y = death_day.deaths.max()
    return y


def highest_case_day(country):
    x = df.loc[(df.Country == country), ['Date_reported', 'Country_code', 'Country', 'WHO_region', 'New_cases',
                                         'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']]
    return x.New_cases.max()


def highest_case_day_world():
    case_day = pd.DataFrame(columns=['date', "cases"])
    for d in df['Date_reported'].unique():
        x = df.loc[(df.Date_reported == d), ['New_cases', 'Date_reported']]
        x = x.New_cases.sum()
        case_day = case_day.append({'date': d, 'cases': x}, ignore_index=True)
    y = case_day.cases.max()
    return y


def percentage_death_world():
    world = df.loc[(df.Date_reported == today), ['Date_reported', 'Cumulative_cases']]
    cases = world.Cumulative_cases.sum()
    world = df.loc[(df.Date_reported == today), ['Date_reported', 'Cumulative_deaths']]
    deaths = world.Cumulative_deaths.sum()
    x = (deaths / cases) * 100
    return ("%.3f" % x) + "%"


def percentage_death(country):
    x = df.loc[(df.Country == country) & (df.Date_reported == today), ['Date_reported', 'Country_code', 'Country',
                                                                       'WHO_region', 'New_cases', 'Cumulative_cases',
                                                                       'New_deaths', 'Cumulative_deaths']]
    country = (x.Cumulative_deaths.sum() / x.Cumulative_cases.sum()) * 100
    return ("%.3f" % country) + "%"


# ------setting up data for testing----------------------
def population_data(country):
    from sklearn.preprocessing import StandardScaler
    from math import sqrt
    population = df.loc[(df.Country == country) & (df.New_deaths > 0), ['New_deaths']]
    values = population.values
    values = values.reshape((len(values), 1))
    scaled_features = StandardScaler().fit_transform(values)
    scaled_features = scaled_features.flatten()
    return scaled_features


# Describe method for population
def describe_population(country):
    population = df.loc[(df.Country == country) & (df.New_deaths > 0), ['New_deaths']]
    return population.describe()


# Distribution filtering for population
def filter_population(country):
    size = len(population_data(country))
    value = population_data(country)
    dist_names = ['beta',
                  'expon',
                  'gamma',
                  'lognorm',
                  'norm',
                  'pearson3',
                  'triang',
                  'uniform',
                  'weibull_min',
                  'weibull_max']
    chi_square = []
    p_values = []

    percentile_bins = np.linspace(0, 100, 51)
    percentile_cutoffs = np.percentile(value, percentile_bins)
    observed_frequency, bins = (np.histogram(value, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    for distribution in dist_names:
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(value)
        p = scipy.stats.kstest(value, distribution, args=param)[1]
        p = np.around(p, 3)
        p_values.append(p)
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2],
                              scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        ss = round(ss, 3)
        chi_square.append(ss)

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    results['p_value'] = p_values
    # results.drop(results[results['p_value'] < .05].index, inplace=True)
    results.sort_values(['chi_square'], inplace=True)

    return results


# sample = 100 days after vaccine
def sample_data(country):
    population = df.loc[(df.Country == country) & (df.New_deaths > 0), ['Date_reported', 'New_deaths']]
    after = population.loc[(population.Date_reported >= '2020-12-08 ') &
                           (population.Date_reported <= today), ['New_deaths']]
    values = after.values
    values = values.reshape((len(values), 1))
    scaled_features = StandardScaler().fit_transform(values)
    scaled_features = scaled_features.flatten()
    return scaled_features


# Describe method for sample
def describe_sample(country):
    population = df.loc[(df.Country == country) & (df.New_deaths > 0), ['Date_reported', 'New_deaths']]
    sample = population.loc[(population.Date_reported >= '2020-12-08 ') &
                            (population.Date_reported <= today), ['New_deaths']]
    return sample.describe()


# Distribution filtering for sample
def filter_sample(country):
    size = len(sample_data(country))
    value = sample_data(country)
    dist_names = ['beta',
                  'expon',
                  'gamma',
                  'lognorm',
                  'norm',
                  'pearson3',
                  'triang',
                  'uniform',
                  'weibull_min',
                  'weibull_max']
    chi_square = []
    p_values = []

    percentile_bins = np.linspace(0, 100, 51)
    percentile_cutoffs = np.percentile(value, percentile_bins)
    observed_frequency, bins = (np.histogram(value, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    for distribution in dist_names:
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(value)
        p = scipy.stats.kstest(value, distribution, args=param)[1]
        p = np.around(p, 3)
        p_values.append(p)
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2],
                              scale=param[-1])
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        ss = round(ss, 3)
        chi_square.append(ss)

    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square
    results['p_value'] = p_values
    # results.drop(results[results['p_value'] < .05].index, inplace=True)
    results.sort_values(['chi_square'], inplace=True)

    return results


def test(country):
    stat, p = mannwhitneyu(population_data(country), sample_data(country))
    alpha = 0.05
    if p > alpha:
        return ('mann whitney u test: \n'
                'Same distribution (fail to reject H0)\n'
                'Statistics=%.3f, p=%.3f' % (stat, p))
    else:
        return ('mann whitney u test: \n'
                'Different distribution (reject H0)\n'
                'Statistics=%.3f, p=%.3f' % (stat, p))


def spapiro_population(country):
    stat, p = shapiro(population_data(country))
    alpha = 0.05
    if p > alpha:
        return ('Shapiro-Wilk test(population) \n'
                'Same distribution (fail to reject H0)\n'
                'Statistics=%.3f, p=%.3f' % (stat, p))

    else:
        return ('Shapiro-Wilk test(population) \n'
                'Different distribution (reject H0)\n'
                'Statistics=%.3f, p=%.3f' % (stat, p))


def spapiro_sample(country):
    stat, p = shapiro(sample_data(country))

    alpha = 0.05
    if p > alpha:
        return ('Shapiro-Wilk test(sample)\n'
                'Same distribution (fail to reject H0)\n'
                'Statistics=%.3f, p=%.3f' % (stat, p))

    else:
        return ('Shapiro-Wilk test(sample) \n'
                'Different distribution (reject H0)\n'
                'Statistics=%.3f, p=%.3f' % (stat, p))


# -----gui--------

app_data = {
    "country": ""
}


class tkinterApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self, height=1000, width=1000, bg="gray")
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, Page1):
            frame = F(container, self)
            self.frames[F] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def onSelectCountry(self, country):
        print("Main Country", country)
        frame = self.frames[Page1]
        frame.onSelectCountry(country)


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="gray")
        self.controller = controller

        style = ttk.Style()
        style.configure('W.TButton', foreground='black', background='white')

        self.country_var = tk.StringVar()
        country_choice = ttk.Combobox(self, width=27, text="pick a country", textvariable=self.country_var)
        country_choice["values"] = ([x for x in df["Country"].unique()])
        country_choice.place(relx=0.14, rely=0.05, relwidth=0.5, relheight=0.07)
        country_choice.current()

        def onSelectCountry(eventObject):
            controller.show_frame(Page1)
            country = self.country_var.get()
            print(country)
            controller.onSelectCountry(country)
            country_choice.delete(0, tk.END)

        country_choice.bind("<<ComboboxSelected>>", onSelectCountry)

        text_box = tk.Text(self, height=28, width = 80, bg="white")
        text_box.insert(tk.INSERT, " \n"
                                   "                         INFORMATION ABOUT THE PROGRAM \n"
                                   "\n"
                                   "Showing the distribution of deaths per day within a country, since the\n"
                                   "first death was reported(Population).then showing distribution of deaths per\n"
                                   "day after the first person was vaccinated on 8/12/2020(Sample). testing if\n"
                                   "data is normally distributed, then testing if the Sample can utilize the same\n"
                                   "distribution as the Population\n"
                                   "\n"
                                   "Population = deaths per-day since the first death from covid\n"
                                   "\n"
                                   "Sample = days after the first person was given the covid vaccine\n"
                                   "\n"
                                   "Tests used: shapiro-wilk, Mann-Whitney U Test\n"
                                   "\n"
                                   "Best used on countries with large populations EX: United States of America,\n"
                                   "The United Kingdom, India \n"
                                   "\n"
                                   "all data is pulled from the World Heath Organization\n"
                                   "link to data: https://covid19.who.int/WHO-COVID-19-global-data.csv\n")
        text_box.config(state="disabled")
        text_box.place(relx=0.14, rely=0.15, )


class Page1(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="gray")
        self.controller = controller

        def clear():
            self.total_deaths_country.delete("1.0", "end")
            self.total_cases_country.delete("1.0", "end")
            self.avg_deaths_perday_country.delete("1.0", "end")
            self.avg_case_pd_country.delete("1.0", "end")
            self.high_death_pd_country.delete("1.0", "end")
            self.high_case_pd_country.delete("1.0", "end")
            self.death_chance_country.delete("1.0", "end")
            self.summary_population.delete("1.0", "end")
            self.summary_sample.delete("1.0", "end")
            self.fliter_population.delete("1.0", "end")
            self.fliter_sample.delete("1.0", "end")
            self.test.delete("1.0", "end")
            self.test0.delete("1.0", "end")
            self.test1.delete("1.0", "end")

        button1 = ttk.Button(self, text="Menu",
                             command=lambda: [controller.show_frame(StartPage), clear()])
        button1.place(relx=0.01, rely=0.01, relwidth=0.1, relheight=0.04)

        self.country_var = tk.StringVar()

        label_country = tk.Label(self, text=app_data["country"], textvariable=self.country_var, bg="white")
        label_country.place(relx=0.38, rely=0.37, relwidth=0.156, relheight=0.03)

        label_world = tk.Label(self, text="world", bg="white")
        label_world.place(relx=0.54, rely=0.37, relwidth=0.0755, relheight=0.03)

        # for total deaths
        label = tk.Label(self, text="total deaths", bg="white")
        label.place(relx=0.38, rely=0.405, relwidth=0.0755, relheight=0.03)
        self.total_deaths_country = tk.Text(self, height=1, width=9, bg="white")
        self.total_deaths_country.place(relx=0.46, rely=0.405)
        total_deaths_world = tk.Text(self, height=1, width=9, bg="white")
        total_deaths_world.insert(tk.INSERT, total_deaths_world_func())
        total_deaths_world.config(state="disabled")
        total_deaths_world.place(relx=0.54, rely=0.405)

        # for total cases
        label = tk.Label(self, text="total cases", bg="white")
        label.place(relx=0.38, rely=0.44, relwidth=0.0755, relheight=0.03)
        self.total_cases_country = tk.Text(self, height=1, width=9, bg="white")
        self.total_cases_country.place(relx=0.46, rely=0.44)
        total_cases_world = tk.Text(self, height=1, width=9, bg="white")
        total_cases_world.insert(tk.INSERT, total_cases_world_func())
        total_cases_world.config(state="disabled")
        total_cases_world.place(relx=0.54, rely=0.44)

        # avg deaths perday
        label = tk.Label(self, text="avg deaths pd", bg="white")
        label.place(relx=0.38, rely=0.475, relwidth=0.0755, relheight=0.03)
        self.avg_deaths_perday_country = tk.Text(self, height=1, width=9, bg="white")
        self.avg_deaths_perday_country.place(relx=0.46, rely=0.475)
        avg_deaths_perday_world = tk.Text(self, height=1, width=9, bg="white")
        avg_deaths_perday_world.insert(tk.INSERT, avg_deaths_pd_world())
        avg_deaths_perday_world.config(state="disabled")
        avg_deaths_perday_world.place(relx=0.54, rely=0.475)

        # avergec case per day
        label = tk.Label(self, text="avg cases pd", bg="white")
        label.place(relx=0.38, rely=0.51, relwidth=0.0755, relheight=0.03)
        self.avg_case_pd_country = tk.Text(self, height=1, width=9, bg="white")
        self.avg_case_pd_country.place(relx=0.46, rely=0.51)
        avg_case_pd_world = tk.Text(self, height=1, width=9, bg="white")
        avg_case_pd_world.insert(tk.INSERT, avg_new_case_pd_world())
        avg_case_pd_world.config(state="disabled")
        avg_case_pd_world.place(relx=0.54, rely=0.51)

        # higest deaths in a day
        label = tk.Label(self, text="most death pd", bg="white")
        label.place(relx=0.38, rely=0.545, relwidth=0.0755, relheight=0.03)
        self.high_death_pd_country = tk.Text(self, height=1, width=9, bg="white")
        self.high_death_pd_country.place(relx=0.46, rely=0.545)
        high_death_pd_world = tk.Text(self, height=1, width=9, bg="white")
        high_death_pd_world.insert(tk.INSERT, highest_death_day_world())
        high_death_pd_world.config(state="disabled")
        high_death_pd_world.place(relx=0.54, rely=0.545)

        # hightest case in a day
        label = tk.Label(self, text="most cases pd", bg="white")
        label.place(relx=0.38, rely=0.58, relwidth=0.0755, relheight=0.03)
        self.high_case_pd_country = tk.Text(self, height=1, width=9, bg="white")
        self.high_case_pd_country.place(relx=0.46, rely=0.58)
        high_case_pd_world = tk.Text(self, height=1, width=9, bg="white")
        high_case_pd_world.insert(tk.INSERT, highest_case_day_world())
        high_case_pd_world.config(state="disabled")
        high_case_pd_world.place(relx=0.54, rely=0.58)

        # persteges chance of death
        label = tk.Label(self, text="chace of death", bg="white")
        label.place(relx=0.38, rely=0.615, relwidth=0.0755, relheight=0.03)
        self.death_chance_country = tk.Text(self, height=1, width=9, bg="white")
        self.death_chance_country.place(relx=0.46, rely=0.615)
        death_chance_world = tk.Text(self, height=1, width=9, bg="white")
        death_chance_world.insert(tk.INSERT, percentage_death_world())
        death_chance_world.config(state="disabled")
        death_chance_world.place(relx=0.54, rely=0.615)

        self.summary_population = tk.Text(self, height=10, width=18, bg="white")
        self.summary_population.place(relx=0.22, rely=0.37)

        self.summary_sample = tk.Text(self, height=10, width=18, bg="white")
        self.summary_sample.place(relx=0.625, rely=0.37)

        self.fliter_population = tk.Text(self, height=11, width=38, bg="white")
        self.fliter_population.place(relx=0.01, rely=0.72)

        self.fliter_sample = tk.Text(self, height=11, width=38, bg="white")
        self.fliter_sample.place(relx=0.73, rely=0.72)

        self.test = tk.Text(self, height=3, width=38, bg="white")
        self.test.place(relx=0.38, rely=0.9)

        self.test0 = tk.Text(self, height=3, width=38, bg="white")
        self.test0.place(relx=0.38, rely=0.82)

        self.test1 = tk.Text(self, height=3, width=38, bg="white")
        self.test1.place(relx=0.38, rely=0.74)

    # --------used for funtions that need a the seleced country-----
    def onSelectCountry(self, country):
        self.country_var.set(country)
        self.total_deaths_country.insert(tk.END, total_deaths(country))
        self.total_cases_country.insert(tk.INSERT, total_cases(country))
        self.avg_deaths_perday_country.insert(tk.INSERT, avg_deaths_pd(country))
        self.avg_case_pd_country.insert(tk.INSERT, avg_new_case_pd(country))
        self.high_death_pd_country.insert(tk.INSERT, highest_death_day(country))
        self.high_case_pd_country.insert(tk.INSERT, highest_case_day(country))
        self.death_chance_country.insert(tk.INSERT, percentage_death(country))

        # --------discriptive-------
        self.summary_population.insert(tk.INSERT, describe_population(country))
        self.summary_sample.insert(tk.INSERT, describe_sample(country))
        self.fliter_population.insert(tk.INSERT, filter_population(country))
        self.fliter_sample.insert(tk.INSERT, filter_sample(country))

        # --------tests--------
        self.test.insert(tk.INSERT, test(country))
        self.test0.insert(tk.INSERT, spapiro_population(country))
        self.test1.insert(tk.INSERT, spapiro_sample(country))

        # -----graphs------
        # population graphs
        figure1 = plt.Figure(figsize=(3, 2), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, self)
        bar1.get_tk_widget().place(relx=0.01, rely=0.06, relwidth=0.48, relheight=0.3)
        ax1.hist(population_data(country), bins=15, density=1, alpha=0.5)
        ax1.set_title('deaths per day (population)')
        ax1.set_xlabel("bin")
        ax1.set_ylabel("count")

        figure10 = plt.Figure(figsize=(3, 2), dpi=100)
        ax10 = figure10.add_subplot(111)
        bar10 = FigureCanvasTkAgg(figure10, self)
        bar10.get_tk_widget().place(relx=0.01, rely=0.37, relwidth=0.2, relheight=0.35)
        sm.qqplot(population_data(country), line='45', ax=ax10)
        ax10.set_title('qq-plot (population)')

        # sample graphs
        figure2 = plt.Figure(figsize=(3, 2), dpi=100)
        ax2 = figure2.add_subplot(111)
        bar2 = FigureCanvasTkAgg(figure2, self)
        bar2.get_tk_widget().place(relx=0.51, rely=0.06, relwidth=0.48, relheight=0.3)
        ax2.hist(sample_data(country), bins=15, density=1, alpha=0.5)
        ax2.set_title('deaths per day (sample)')
        ax2.set_xlabel("bin")
        ax2.set_ylabel("count")

        figure20 = plt.Figure(figsize=(3, 2), dpi=100)
        ax20 = figure20.add_subplot(111)
        bar20 = FigureCanvasTkAgg(figure20, self)
        bar20.get_tk_widget().place(relx=.79, rely=0.37, relwidth=0.2, relheight=0.35)
        sm.qqplot(sample_data(country), line='45', ax=ax20)
        ax20.set_title('qq-plot(sample)')


app = tkinterApp()
app.mainloop()
