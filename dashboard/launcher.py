#!/usr/bin/env python3
"""
Launcher for OpsNexus-ML Dashboard
Provides both text-based and web-based dashboard options
"""

import sys
import os
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

def launch_text_dashboard():
    """Launch the text-based dashboard in the terminal"""
    print("🚀 Launching text-based dashboard...")
    try:
        subprocess.run([sys.executable, "realtime_dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Text dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching text dashboard: {e}")

def launch_web_dashboard():
    """Launch the web-based dashboard"""
    print("🌐 Launching web-based dashboard...")
    try:
        # Start a simple HTTP server to serve the dashboard
        dashboard_dir = Path(__file__).parent.absolute()
        os.chdir(dashboard_dir)

        # Try to open browser automatically
        def open_browser():
            time.sleep(2)  # Wait for server to start
            webbrowser.open('http://localhost:8080')

        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()

        # Start HTTP server
        subprocess.run([sys.executable, "-m", "http.server", "8080"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Web dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching web dashboard: {e}")

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🚀 OpsNexus-ML Dashboard Launcher")
    print("=" * 60)
    print("Choose dashboard type:")
    print("  1. Text-based dashboard (runs in terminal)")
    print("  2. Web-based dashboard (opens in browser)")
    print("  3. Both (text in terminal, web in browser)")
    print("  4. Exit")
    print()

    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
                launch_text_dashboard()
                break
            elif choice == "2":
                launch_web_dashboard()
                break
            elif choice == "3":
                print("🚀 Launching both dashboards...")
                # Launch text dashboard in background thread
                text_thread = threading.Thread(target=launch_text_dashboard)
                text_thread.daemon = True
                text_thread.start()

                # Launch web dashboard
                launch_web_dashboard()
                break
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n👋 Launcher interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Change to dashboard directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    main()