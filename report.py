"""
report.py: which days the raw stores hold and how complete they are.

  python report.py --config config.json            # summary and the days needing attention
  python report.py --config config.json --all      # every day
  python report.py --config config.json --csv days.csv
"""

if __name__ == "__main__":
    from goes_downloader.report import main

    main()
