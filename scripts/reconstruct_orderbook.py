import os
import sys
# Add root directory to sys.path to allow importing from 'models'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import numpy as np
import glob
from ohlcv_to_orderbook import OrderbookGenerator
from ohlcv_to_orderbook.config import OrderbookConfig

# Configuration
def reconstruct_orderbooks(input_base, output_base):
    COLS_12 = ['open_ts_ms', 'o', 'h', 'l', 'c', 'v',
               'close_t', 'quote_v', 'n_trades',
               'taker_buy_base', 'taker_buy_quote', 'ignore']
    COLS_6  = ['open_ts_ms', 'o', 'h', 'l', 'c', 'v']

    def load_ohlcv_file(path):
        peek = pd.read_csv(path, nrows=1, header=None)
        try:
            float(str(peek.iloc[0, 0]))
            headerless = True
        except ValueError:
            headerless = False
        if headerless:
            df = pd.read_csv(path, header=None)
            df.columns = COLS_12 if df.shape[1] == 12 else COLS_6
        else:
            df = pd.read_csv(path)
        return df

    # Find all OHLCV source files — processed, synthetic, raw 1m, and klines
    processed_files = glob.glob(os.path.join(input_base, "**/*.csv"), recursive=True)
    processed_files = [
        f for f in processed_files
        if (f.endswith('_processed.csv') and '_processed_processed' not in f)
        or f.endswith('_synthetic.csv')
        or '-1m-' in f
        or f.endswith('_klines.csv')
    ]
    # Skip already-reconstructed L2 files landing in the source tree
    processed_files = [f for f in processed_files if '_L2.csv' not in f]
    print(f"Found {len(processed_files)} target files in {input_base} for reconstruction.")

    for file_path in processed_files:
        print(f"Reconstructing orderbook for {file_path}...")
        try:
            df = load_ohlcv_file(file_path)

            # Map columns for ohlcv-to-orderbook
            if 'open_ts_ms' in df.columns:
                ts = df['open_ts_ms'] // 1000
            else:
                ts = np.arange(len(df)) * 60
                
            ohlcv_df = pd.DataFrame({
                'timestamp': ts,
                'open': df['o'],
                'high': df['h'],
                'low': df['l'],
                'close': df['c'],
                'volume': df['v']
            })
            
            # Initialize config and generator
            config = OrderbookConfig(spread_percentage=0.001)
            generator = OrderbookGenerator(config=config)
            
            # Generate synthetic orderbook snapshots
            orderbook_snapshots = generator.generate_orderbook_data(ohlcv_df)
            
            # Strict schema enforcement: reconstructed/[SOURCE_TYPE]/[EXCHANGE]/[REGIME]/[SYMBOL]
            normalized_path = file_path.replace("\\", "/")
            parts = normalized_path.split('/')
            
            # Identify where 'real' or 'synthetic' starts
            try:
                base_idx = -1
                source_type = "unknown"
                for i, part in enumerate(parts):
                    if part in ['real', 'synthetic']:
                        base_idx = i
                        source_type = part
                        break
                
                if base_idx == -1:
                     raise ValueError(f"Could not determine source type (real/synthetic) from path {normalized_path}")
                
                exchange = parts[base_idx + 1]
                regime = parts[base_idx + 2]
                symbol = parts[base_idx + 3]
                
                target_dir = os.path.join("reconstructed", exchange, regime, symbol)
                os.makedirs(target_dir, exist_ok=True)
            except Exception as e:
                print(f"Skipping {file_path} - does not match [TYPE]/[EXCH]/[REGIME]/[SYMB] schema: {e}")
                continue
            
            # Save snapshots
            # Ensure we always add _L2 suffix and keep .csv
            original_basename = os.path.basename(file_path)
            if original_basename.endswith(".csv"):
                filename = original_basename[:-4] + "_L2.csv"
            else:
                filename = original_basename + "_L2.csv"
                
            output_file = os.path.join(target_dir, filename)
            
            if isinstance(orderbook_snapshots, pd.DataFrame):
                orderbook_snapshots.to_csv(output_file, index=False)
            else:
                pd.DataFrame(orderbook_snapshots).to_csv(output_file, index=False)

            print(f"Successfully saved reconstructed L2 to {output_file}")

            # Save trade-flow file derived from OHLCV taker buy/sell volumes
            # Only possible when the source has the 12-column Binance format
            if 'taker_buy_base' in df.columns and 'v' in df.columns:
                try:
                    trades_rows = []
                    for ts_idx, row in enumerate(df.itertuples(index=False)):
                        total_vol = float(row.v) if row.v and row.v == row.v else 0.0
                        buy_vol   = float(row.taker_buy_base) if (row.taker_buy_base is not None and row.taker_buy_base == row.taker_buy_base) else 0.0
                        sell_vol  = max(0.0, total_vol - buy_vol)
                        mid_price = (float(row.h) + float(row.l)) / 2.0
                        if buy_vol > 0:
                            trades_rows.append({'timestamp_idx': ts_idx, 'side': 'buy',
                                                'price': mid_price, 'size': buy_vol})
                        if sell_vol > 0:
                            trades_rows.append({'timestamp_idx': ts_idx, 'side': 'sell',
                                                'price': mid_price, 'size': sell_vol})
                    if trades_rows:
                        trades_file = output_file.replace("_L2.csv", "_trades.csv")
                        pd.DataFrame(trades_rows).to_csv(trades_file, index=False)
                except Exception:
                    pass
            
        except Exception as e:
            print(f"Error reconstructing {file_path}: {e}")

def main():
    # Scan all Real data
    reconstruct_orderbooks("real", "reconstructed")
    # Scan all Synthetic data
    reconstruct_orderbooks("synthetic", "reconstructed")

if __name__ == "__main__":
    main()
