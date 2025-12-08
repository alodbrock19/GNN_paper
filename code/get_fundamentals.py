import yfinance as yf
import pandas as pd
import numpy as np

def get_rich_fundamentals(ticker_symbol, start_date="2020-01-01"):
    print(f"Processing {ticker_symbol}...")
    
    # 1. Fetch Data
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # A. Price History
        prices = ticker.history(start=start_date)[['Close', 'Volume']]
        if prices.empty:
            print(f"Error: No price data available for {ticker_symbol}")
            return None
        prices.index = prices.index.tz_localize(None)

        # B. Benchmark (SPY) for Beta
        # Using a try-except block specifically for SPY to avoid crashing the whole script
        try:
            spy = yf.Ticker("SPY").history(start=start_date)['Close'].rename("SPY_Close")
            spy.index = spy.index.tz_localize(None)
        except:
            spy = pd.Series(dtype=float)

        # C. Quarterly Financials & Dividends
        # Transpose so dates are the index
        fin = ticker.quarterly_financials.T
        bs = ticker.quarterly_balance_sheet.T
        cashflow = ticker.quarterly_cash_flow.T 
        dividends = ticker.dividends 
        
        # Check if financial data is available
        if fin.empty or bs.empty:
            print(f"Warning: Empty financial data for {ticker_symbol}")
            # Create empty DataFrames with expected structure
            fin = pd.DataFrame()
            bs = pd.DataFrame()
        
        # --- FIX 2: True TTM Calculation (Rolling 4 Quarters) ---
        # We calculate the sum of the last 4 quarters BEFORE merging to daily data
        # This fixes the seasonality issue.
        if not fin.empty:
            fin = fin.sort_index() # Ensure chronological order
            
            # Find Net Income column (try common variations)
            net_income_col = None
            for col_name in ['Net Income', 'NetIncome', 'Net income', 'NetIncomeCommonStockholders']:
                if col_name in fin.columns:
                    net_income_col = col_name
                    break
            
            # Find Total Revenue column (try common variations)
            revenue_col = None
            for col_name in ['Total Revenue', 'TotalRevenue', 'Revenue', 'Total revenue']:
                if col_name in fin.columns:
                    revenue_col = col_name
                    break
            
            # Calculate TTM values if columns exist
            if net_income_col:
                fin['Net_Income_TTM'] = fin[net_income_col].rolling(window=4, min_periods=1).sum()
            else:
                fin['Net_Income_TTM'] = np.nan
                print(f"  Warning: Net Income column not found for {ticker_symbol}")
            
            if revenue_col:
                fin['Revenue_TTM'] = fin[revenue_col].rolling(window=4, min_periods=1).sum()
            else:
                fin['Revenue_TTM'] = np.nan
                print(f"  Warning: Total Revenue column not found for {ticker_symbol}")
        else:
            # Create empty TTM columns if no financial data
            fin = pd.DataFrame()
            fin['Net_Income_TTM'] = pd.Series(dtype=float)
            fin['Revenue_TTM'] = pd.Series(dtype=float)
        
        # Handle Dividends (Rolling 12 Month Sum)
        if not dividends.empty:
            dividends.index = dividends.index.tz_localize(None)
            dividends_rolling = dividends.rolling(window=365, min_periods=1).sum().rename("Dividends_TTM")
        else:
            dividends_rolling = pd.Series(dtype=float, name="Dividends_TTM")

    except Exception as e:
        print(f"Error fetching fundamental data: {e}")
        return None

    # 2. Merge Data
    # Merge Financials and Balance Sheet
    if not fin.empty and not bs.empty:
        reports = fin.join(bs, lsuffix='_fin', rsuffix='_bs', how='outer')
    elif not fin.empty:
        reports = fin.copy()
    elif not bs.empty:
        reports = bs.copy()
    else:
        # Both are empty, create empty DataFrame with TTM columns
        reports = pd.DataFrame()
        reports['Net_Income_TTM'] = pd.Series(dtype=float)
        reports['Revenue_TTM'] = pd.Series(dtype=float)
    
    if not reports.empty:
        reports.index = pd.to_datetime(reports.index).tz_localize(None)
    
    # --- FIX 3: Look-Ahead Bias Mitigation (Optional but Recommended) ---
    # Shift reports forward by ~45 days to approximate earnings release lag
    # reports.index = reports.index + pd.Timedelta(days=45) 

    # Merge everything into one daily dataframe
    df = prices.join(reports, how='outer')
    if not spy.empty:
        df = df.join(spy, how='outer')
    else:
        df['SPY_Close'] = np.nan
    
    if not dividends_rolling.empty:
        df = df.join(dividends_rolling, how='outer')
    else:
        df['Dividends_TTM'] = np.nan
    
    df = df.sort_index().ffill() # Forward fill creates the "Point-in-Time" view
    df = df.loc[start_date:]     # Crop to start date AFTER filling

    # 3. Calculate Derived Fundamentals
    
    # A. Shares & Market Cap
    try:
        shares = ticker.get_shares_full(start=start_date)
        shares = pd.DataFrame({'Shares': shares})
        shares.index = shares.index.tz_localize(None)
        df = df.join(shares, how='outer').sort_index().ffill().loc[start_date:]
    except:
        df['Shares'] = np.nan

    df['marketCap'] = df['Close'] * df['Shares']

    # B. Valuation Metrics (Using TTM values now)
    # Net Income TTM is already calculated correctly from the rolling window above
    # Handle missing TTM columns gracefully
    net_income_ttm = df['Net_Income_TTM'] if 'Net_Income_TTM' in df.columns else pd.Series(np.nan, index=df.index)
    revenue_ttm = df['Revenue_TTM'] if 'Revenue_TTM' in df.columns else pd.Series(np.nan, index=df.index)
    dividends_ttm = df['Dividends_TTM'] if 'Dividends_TTM' in df.columns else pd.Series(np.nan, index=df.index)

    df['trailingEps'] = np.where(
        (df['Shares'] > 0) & (~net_income_ttm.isna()),
        net_income_ttm / df['Shares'], 
        np.nan
    )
    
    df['trailingPE'] = np.where(
        (df['trailingEps'] > 0) & (df['Close'] > 0), 
        df['Close'] / df['trailingEps'], 
        np.nan
    )

    # --- FIX 1: Payout Ratio Calculation ---
    # Dividend Yield and Payout Ratio
    df['payoutRatio'] = np.where(
        (net_income_ttm > 0) & (~dividends_ttm.isna()), 
        dividends_ttm / net_income_ttm, 
        np.nan
    )
    
    # C. Book Value & Price to Book
    # Fix: Use conditional column access instead of df.get()
    assets = df['Total Assets'] if 'Total Assets' in df.columns else pd.Series(np.nan, index=df.index)
    
    # Try common variations of liabilities column name
    liabs = None
    for col_name in ['Total Liabilities Net Minority Interest', 'Total Liabilities', 
                     'TotalLiabilitiesNetMinorityInterest', 'Liabilities']:
        if col_name in df.columns:
            liabs = df[col_name]
            break
    if liabs is None:
        liabs = pd.Series(np.nan, index=df.index)
    
    df['bookValue'] = assets - liabs
    
    df['priceToBook'] = np.where(
        (df['bookValue'] > 0) & (~df['bookValue'].isna()), 
        df['marketCap'] / df['bookValue'], 
        np.nan
    )

    # D. Margins
    df['profitMargins'] = np.where(
        (revenue_ttm > 0) & (~revenue_ttm.isna()) & (~net_income_ttm.isna()),
        net_income_ttm / revenue_ttm, 
        np.nan
    )

    # E. Enterprise Value
    # Fix: Use conditional column access instead of df.get()
    debt = df['Total Debt'] if 'Total Debt' in df.columns else pd.Series(0, index=df.index)
    
    # Try common variations of cash column name
    cash = None
    for col_name in ['Cash And Cash Equivalents', 'Cash and Cash Equivalents', 
                     'CashAndCashEquivalents', 'Cash']:
        if col_name in df.columns:
            cash = df[col_name]
            break
    if cash is None:
        cash = pd.Series(0, index=df.index)
    
    enterprise_value = df['marketCap'] + debt - cash
    
    df['enterpriseToRevenue'] = np.where(
        (revenue_ttm > 0) & (~revenue_ttm.isna()) & (~enterprise_value.isna()),
        enterprise_value / revenue_ttm, 
        np.nan
    )

    # 4. Technicals
    df['52WeekChange'] = df['Close'].pct_change(periods=252)
    df['averageVolume'] = df['Volume'].rolling(window=30).mean()

    # Beta Calculation
    # Beta = Cov(Stock, Market) / Var(Market) = Correlation * (Std(Stock) / Std(Market))
    if 'SPY_Close' in df.columns and not df['SPY_Close'].isna().all():
        returns_stock = df['Close'].pct_change()
        returns_market = df['SPY_Close'].pct_change()
        
        # Calculate rolling correlation and standard deviations
        rolling_corr = returns_stock.rolling(window=126).corr(returns_market)
        rolling_std_stock = returns_stock.rolling(window=126).std()
        rolling_std_market = returns_market.rolling(window=126).std()
        
        # Beta = Correlation * (Std(Stock) / Std(Market))
        # This is mathematically equivalent to Cov(Stock, Market) / Var(Market)
        df['beta'] = np.where(
            (rolling_std_market > 0) & (~rolling_corr.isna()) & (~rolling_std_stock.isna()),
            rolling_corr * (rolling_std_stock / rolling_std_market),
            np.nan
        )
    else:
        df['beta'] = np.nan

    # 5. Final Output
    final_cols = [
        "marketCap", "trailingPE", "priceToBook", "trailingEps", 
        "bookValue", "payoutRatio", "beta", "52WeekChange", 
        "averageVolume", "enterpriseToRevenue", "profitMargins"
    ]
    
    available_cols = [c for c in final_cols if c in df.columns]
    return df[available_cols].dropna(how='all') # Drop rows where ALL fundamentals are NaN

# --- Run Test ---
# df = get_rich_fundamentals("NVDA")
# print(df.tail())