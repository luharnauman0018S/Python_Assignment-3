import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

class CovidDashboard:
    def __init__(self):
        # Create data folder if it doesn't exist
        self.data_dir = 'covid_data'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Data sources
        self.global_data_url = 'https://disease.sh/v3/covid-19/all'
        self.countries_data_url = 'https://disease.sh/v3/covid-19/countries'
        self.historical_data_url = 'https://disease.sh/v3/covid-19/historical/all?lastdays=30'
        
        # Initialize data storage
        self.global_data = None
        self.countries_data = None
        self.historical_data = None
        self.top_countries = None
        
    def fetch_data(self):
        """Fetch COVID-19 data from API and save locally"""
        # Fetch global summary
        response = requests.get(self.global_data_url)
        self.global_data = response.json()
        with open(f'{self.data_dir}/global_summary.json', 'w') as f:
            json.dump(self.global_data, f)
        
        # Fetch countries data
        response = requests.get(self.countries_data_url)
        self.countries_data = response.json()
        with open(f'{self.data_dir}/countries_data.json', 'w') as f:
            json.dump(self.countries_data, f)
        
        # Create pandas DataFrame for countries
        df_countries = pd.DataFrame(self.countries_data)
        self.countries_df = df_countries
        df_countries.to_csv(f'{self.data_dir}/countries_data.csv', index=False)
        
        # Fetch historical data
        response = requests.get(self.historical_data_url)
        self.historical_data = response.json()
        with open(f'{self.data_dir}/historical_data.json', 'w') as f:
            json.dump(self.historical_data, f)
            
        print(f"Data fetched and saved to {self.data_dir}/")
        
    def process_data(self):
        """Process raw data for visualization"""
        if self.countries_data is None:
            self.load_data()
            
        # Process countries data
        df = pd.DataFrame(self.countries_data)
        
        # Get top 10 countries by cases
        self.top_countries = df.sort_values('cases', ascending=False).head(10)
        self.top_countries.to_csv(f'{self.data_dir}/top_countries.csv', index=False)
        
        # Process historical data
        if self.historical_data is not None:
            # Convert historical data to DataFrame
            cases = self.historical_data['cases']
            deaths = self.historical_data['deaths']
            recovered = self.historical_data.get('recovered', {})
            
            # Create timeseries DataFrames
            self.cases_df = pd.DataFrame.from_dict(cases, orient='index', columns=['cases'])
            self.deaths_df = pd.DataFrame.from_dict(deaths, orient='index', columns=['deaths'])
            
            if recovered:
                self.recovered_df = pd.DataFrame.from_dict(recovered, orient='index', columns=['recovered'])
                self.recovered_df.to_csv(f'{self.data_dir}/historical_recovered.csv')
            
            self.cases_df.to_csv(f'{self.data_dir}/historical_cases.csv')
            self.deaths_df.to_csv(f'{self.data_dir}/historical_deaths.csv')
            
    def load_data(self):
        """Load data from local files if available"""
        try:
            with open(f'{self.data_dir}/global_summary.json', 'r') as f:
                self.global_data = json.load(f)
                
            with open(f'{self.data_dir}/countries_data.json', 'r') as f:
                self.countries_data = json.load(f)
                
            with open(f'{self.data_dir}/historical_data.json', 'r') as f:
                self.historical_data = json.load(f)
                
            print("Data loaded from local files")
        except FileNotFoundError:
            print("Local data not found. Please fetch data first.")
            
    def visualize_data(self):
        """Create visualizations for COVID-19 data"""
        if self.countries_data is None or self.global_data is None:
            print("No data available. Please fetch or load data first.")
            return
            
        # Set style
        sns.set(style="whitegrid")
        plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2)
        
        # 1. Global summary
        ax1 = plt.subplot(gs[0, 0])
        global_metrics = ['cases', 'deaths', 'recovered', 'active']
        values = [self.global_data[metric] for metric in global_metrics]
        ax1.bar(global_metrics, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax1.set_title('Global COVID-19 Summary', fontsize=14)
        for i, v in enumerate(values):
            ax1.text(i, v * 0.9, f"{v:,}", ha='center', fontsize=10)
        
        # 2. Top 10 countries by cases
        ax2 = plt.subplot(gs[0, 1])
        countries = self.top_countries['country'].values
        cases = self.top_countries['cases'].values
        ax2.barh(countries[::-1], cases[::-1], color='#3498db')
        ax2.set_title('Top 10 Countries by Cases', fontsize=14)
        for i, v in enumerate(cases[::-1]):
            ax2.text(v * 0.6, i, f"{v:,}", va='center', fontsize=9)
        
        # 3. Cases vs. Deaths for top 10 countries
        ax3 = plt.subplot(gs[1, 0])
        deaths = self.top_countries['deaths'].values
        ax3.scatter(cases, deaths, s=cases/deaths*10, alpha=0.7)
        for i, country in enumerate(countries):
            ax3.annotate(country, (cases[i], deaths[i]))
        ax3.set_xlabel('Total Cases')
        ax3.set_ylabel('Total Deaths')
        ax3.set_title('Cases vs Deaths (bubble size = case/death ratio)', fontsize=14)
        
        # 4. Historical data trend
        ax4 = plt.subplot(gs[1, 1])
        if hasattr(self, 'cases_df'):
            dates = pd.to_datetime(self.cases_df.index)
            ax4.plot(dates, self.cases_df['cases'], label='Cases', color='#3498db')
            ax4.plot(dates, self.deaths_df['deaths'], label='Deaths', color='#e74c3c')
            ax4.set_title('COVID-19 Trend (Last 30 Days)', fontsize=14)
            ax4.legend()
            # Format x-axis dates
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax4.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            plt.xticks(rotation=45)
        
        # 5. Recovery rate by top countries
        ax5 = plt.subplot(gs[2, 0])
        if 'recovered' in self.top_countries.columns:
            recovery_rate = (self.top_countries['recovered'] / self.top_countries['cases'] * 100).values
            ax5.barh(countries[::-1], recovery_rate[::-1], color='#2ecc71')
            ax5.set_title('Recovery Rate by Country (%)', fontsize=14)
            ax5.set_xlim(0, 100)
            for i, v in enumerate(recovery_rate[::-1]):
                ax5.text(v + 2, i, f"{v:.1f}%", va='center', fontsize=9)
        
        # 6. Active vs Recovered proportion
        ax6 = plt.subplot(gs[2, 1])
        if 'recovered' in self.top_countries.columns and 'active' in self.top_countries.columns:
            recovered = self.top_countries['recovered'].values
            active = self.top_countries['active'].values
            total = recovered + active
            recovered_pct = recovered / total * 100
            active_pct = active / total * 100
            
            width = 0.8
            ax6.barh(countries[::-1], recovered_pct[::-1], width, label='Recovered', color='#2ecc71')
            ax6.barh(countries[::-1], active_pct[::-1], width, left=recovered_pct[::-1], label='Active', color='#f39c12')
            ax6.set_title('Active vs Recovered Cases (%)', fontsize=14)
            ax6.legend(loc='lower right')
            ax6.set_xlim(0, 100)
        
        # Add timestamp
        plt.figtext(0.5, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                   ha="center", fontsize=10, style='italic')
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f'{self.data_dir}/covid_dashboard.png', dpi=300)
        plt.savefig(f'{self.data_dir}/covid_dashboard.pdf')
        print(f"Visualizations saved to {self.data_dir}/")
        plt.show()
        
    def generate_report(self):
        """Generate a text report summarizing key findings"""
        if self.global_data is None:
            print("No data available. Please fetch or load data first.")
            return
            
        report = "COVID-19 PANDEMIC SUMMARY REPORT\n"
        report += "=" * 30 + "\n\n"
        report += f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        # Global stats
        report += "GLOBAL STATISTICS\n"
        report += "-" * 20 + "\n"
        report += f"Total Cases: {self.global_data['cases']:,}\n"
        report += f"Total Deaths: {self.global_data['deaths']:,}\n"
        report += f"Total Recovered: {self.global_data['recovered']:,}\n"
        report += f"Active Cases: {self.global_data['active']:,}\n"
        report += f"Critical Cases: {self.global_data.get('critical', 'N/A'):,}\n"
        report += f"Cases Today: {self.global_data.get('todayCases', 'N/A'):,}\n"
        report += f"Deaths Today: {self.global_data.get('todayDeaths', 'N/A'):,}\n\n"
        
        # Global rates
        death_rate = (self.global_data['deaths'] / self.global_data['cases']) * 100
        recovery_rate = (self.global_data['recovered'] / self.global_data['cases']) * 100
        report += f"Global Death Rate: {death_rate:.2f}%\n"
        report += f"Global Recovery Rate: {recovery_rate:.2f}%\n\n"
        
        # Top countries
        report += "TOP 10 COUNTRIES BY CASES\n"
        report += "-" * 30 + "\n"
        report += f"{'Country':<15} {'Cases':>12} {'Deaths':>10} {'Recovery Rate':>15}\n"
        
        for _, row in self.top_countries.iterrows():
            country = row['country']
            cases = row['cases']
            deaths = row['deaths']
            if 'recovered' in row:
                recovery = (row['recovered'] / cases * 100)
                recovery_str = f"{recovery:.1f}%"
            else:
                recovery_str = "N/A"
                
            report += f"{country:<15} {cases:>12,} {deaths:>10,} {recovery_str:>15}\n"
        
        # Save report
        with open(f'{self.data_dir}/covid_report.txt', 'w') as f:
            f.write(report)
            
        print(f"Report saved to {self.data_dir}/covid_report.txt")
        return report
            
def main():
    print("COVID-19 Data Analysis Dashboard")
    print("=" * 30)
    
    dashboard = CovidDashboard()
    
    while True:
        print("\nOptions:")
        print("1. Fetch latest COVID-19 data")
        print("2. Load data from local files")
        print("3. Process data")
        print("4. Visualize data")
        print("5. Generate report")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            dashboard.fetch_data()
        elif choice == '2':
            dashboard.load_data()
        elif choice == '3':
            dashboard.process_data()
        elif choice == '4':
            dashboard.visualize_data()
        elif choice == '5':
            report = dashboard.generate_report()
            print("\nREPORT PREVIEW:")
            print(report[:500] + "...\n")
        elif choice == '6':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
