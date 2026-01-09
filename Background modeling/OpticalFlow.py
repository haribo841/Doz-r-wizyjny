import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import time
import csv

# Konfiguracja backendu matplotlib (tryb bezokienkowy dla stabilności)
matplotlib.use('Agg')

class OpticalFlowSuite:
    def __init__(self, video_filename='Highway.mp4'):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_path = os.path.join(self.script_dir, video_filename)
        self.output_folder = os.path.join(self.script_dir, "optical_flow_results")
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        print(f"--- INICJALIZACJA SUITE ---")
        print(f"Wideo źródłowe: {self.video_path}")
        print(f"Folder wyników: {self.output_folder}")
        
        if not os.path.exists(self.video_path):
            print(f"BŁĄD: Nie znaleziono pliku {video_filename}. Upewnij się, że jest w tym samym folderze co skrypt.")
            # Próba fallbacku na inne popularne nazwy z przesłanych plików
            alternatives = ['people.mp4', 'people2.mp4']
            found = False
            for alt in alternatives:
                alt_path = os.path.join(self.script_dir, alt)
                if os.path.exists(alt_path):
                    self.video_path = alt_path
                    print(f"Znaleziono alternatywę: {alt}")
                    found = True
                    break
            if not found:
                raise FileNotFoundError("Brak pliku wideo.")

    # --- METODY POMOCNICZE (HELPERY) ---

    def _draw_optical_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step//2:h:step, step//2:w:step].astype(np.int32)
        x, y = x.flatten(), y.flatten()
        fx, fy = flow[y, x, 0], flow[y, x, 1]
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (x1, y1), (x2, y2) in lines:
            cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
        return vis

    def _draw_directional_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step//2:h:step, step//2:w:step].astype(np.int32)
        x, y = x.flatten(), y.flatten()
        fx, fy = flow[y, x, 0], flow[y, x, 1]
        magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Kolory dla 8 kierunków (BGR)
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0)
        ]
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)

        for (x1, y1), (x2, y2), ang, mag in zip(lines[:, 0], lines[:, 1], angle, magnitude):
            if mag > 1.0:
                color_idx = int(ang // 45) % 8
                cv2.arrowedLine(vis, (x1, y1), (x2, y2), colors[color_idx], 1, tipLength=0.3)
        return vis

    def _get_heatmap(self, flow):
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        heatmap = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # --- ZADANIE 1: GENEROWANIE WIDEO (FLOW, HEATMAP, DIRECTION) ---
    def run_video_generation(self):
        print("\n[1/5] Generowanie wideo z wynikami (Vectors, Heatmap, Directions)...")
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vec = cv2.VideoWriter(os.path.join(self.output_folder, "vectors.avi"), fourcc, fps, (width, height))
        out_heat = cv2.VideoWriter(os.path.join(self.output_folder, "heatmap.avi"), fourcc, fps, (width, height))
        out_dir = cv2.VideoWriter(os.path.join(self.output_folder, "directions.avi"), fourcc, fps, (width, height))

        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        frame_count = 0
        max_frames = 200 # Ograniczenie dla szybkości testu, usuń limit dla pełnego wideo
        
        while ret and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                                pyr_scale=0.5, levels=3, winsize=15, 
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            
            # 1. Vectors
            out_vec.write(self._draw_optical_flow(gray, flow))
            # 2. Heatmap
            out_heat.write(self._get_heatmap(flow))
            # 3. Directions
            out_dir.write(self._draw_directional_flow(gray, flow))
            
            prev_gray = gray
            frame_count += 1
            if frame_count % 50 == 0: print(f"    Przetworzono {frame_count} klatek...")

        cap.release()
        out_vec.release()
        out_heat.release()
        out_dir.release()
        print("    Zapisano pliki .avi w folderze wyników.")

    # --- ZADANIE 2: PORÓWNANIE PARAMETRÓW ---
    def run_parameter_comparison(self, target_frame=100):
        print(f"\n[2/5] Porównanie parametrów algorytmu (Klatka {target_frame})...")
        experiments = {
            "1_Default":          dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0),
            "2_LargeWin_Smooth":  dict(pyr_scale=0.5, levels=3, winsize=50, iterations=3, poly_n=7, poly_sigma=1.5, flags=0),
            "3_SmallWin_Detail":  dict(pyr_scale=0.5, levels=3, winsize=5,  iterations=3, poly_n=5, poly_sigma=1.1, flags=0),
            "4_More_Levels":      dict(pyr_scale=0.5, levels=7, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0),
            "5_More_Iterations":  dict(pyr_scale=0.5, levels=3, winsize=15, iterations=10, poly_n=5, poly_sigma=1.2, flags=0)
        }
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
        ret, prev = cap.read()
        ret, curr = cap.read()
        cap.release()
        
        if not ret: return

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        for name, params in experiments.items():
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **params)
            vec_img = self._draw_optical_flow(gray, flow)
            heat_img = self._get_heatmap(flow)
            combined = np.hstack((vec_img, heat_img))
            
            cv2.putText(combined, f"Config: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(self.output_folder, f"param_compare_{name}.png"), combined)
        print("    Zapisano obrazy porównawcze.")

    # --- ZADANIE 3: WPŁYW ROZDZIELCZOŚCI ---
    def run_resolution_analysis(self, target_frame=100):
        print(f"\n[3/5] Analiza wpływu rozdzielczości...")
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
        ret, prev = cap.read()
        ret, curr = cap.read()
        cap.release()
        
        if not ret: return
        
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        h, w = prev_gray.shape

        scales = [0.25, 0.5, 1.0]
        results = []
        times = []
        labels = []

        for scale in scales:
            new_dim = (int(w * scale), int(h * scale))
            p_resized = cv2.resize(prev_gray, new_dim)
            c_resized = cv2.resize(curr_gray, new_dim)
            
            start = time.time()
            flow = cv2.calcOpticalFlowFarneback(p_resized, c_resized, None, 
                                                pyr_scale=0.5, levels=3, winsize=15, 
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            dur = (time.time() - start) * 1000
            
            # Korekta skali do wizualizacji
            flow_corr = flow * (1.0 / scale)
            mag, _ = cv2.cartToPolar(flow_corr[..., 0], flow_corr[..., 1])
            mag_upscaled = cv2.resize(mag, (w, h), interpolation=cv2.INTER_NEAREST)
            
            results.append(mag_upscaled)
            times.append(dur)
            labels.append(f"{new_dim}\n({scale})")

        # Wizualizacja
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Resolution Impact', fontsize=16)
        global_max = max([np.max(m) for m in results])

        for i, (mag, p_time, label) in enumerate(zip(results, times, labels)):
            axes[0, i].imshow(mag, cmap='jet', vmin=0, vmax=global_max)
            axes[0, i].set_title(label)
            axes[0, i].axis('off')
            
            axes[1, i].bar(['Time [ms]'], [p_time], color='orange')
            axes[1, i].text(0, p_time, f"{p_time:.1f}ms", ha='center', va='bottom')
            axes[1, i].set_ylim(0, max(times)*1.2)

        plt.savefig(os.path.join(self.output_folder, "resolution_comparison.png"))
        plt.close()
        print("    Zapisano wykres resolution_comparison.png.")

    # --- ZADANIE 4: STATYSTYKA I WYKRESY ---
    def run_statistics_and_plots(self):
        print("\n[4/5] Analiza statystyczna i generowanie CSV...")
        csv_filename = "motion_statistics.csv"
        csv_path = os.path.join(self.output_folder, csv_filename)
        
        cap = cv2.VideoCapture(self.video_path)
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Grid settings
        grid_size = (4, 4)
        h_img, w_img = prev_gray.shape
        rh, rw = h_img // grid_size[0], w_img // grid_size[1]

        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Frame', 'Row', 'Col', 'Avg_Intensity']
            writer.writerow(header)
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                                    pyr_scale=0.5, levels=3, winsize=15, 
                                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Zapis statystyk dla siatki 4x4
                for i in range(grid_size[0]):
                    for j in range(grid_size[1]):
                        region = mag[i*rh:(i+1)*rh, j*rw:(j+1)*rw]
                        avg_int = np.mean(region)
                        writer.writerow([frame_idx, i, j, avg_int])
                
                prev_gray = gray
                frame_idx += 1
                if frame_idx % 100 == 0: print(f"    Statystyka: klatka {frame_idx}...")
                
        cap.release()
        print(f"    Plik CSV zapisany: {csv_path}")
        
        # --- CZĘŚĆ PLOTTING ---
        print("\n[5/5] Generowanie wykresu na podstawie danych...")
        df = pd.read_csv(csv_path)
        
        # Znalezienie najaktywniejszego obszaru
        activity_sum = df.groupby(['Row', 'Col'])['Avg_Intensity'].sum()
        best_area = activity_sum.idxmax()
        best_row, best_col = best_area
        
        area_data = df[(df['Row'] == best_row) & (df['Col'] == best_col)].sort_values('Frame')
        
        plt.figure(figsize=(10, 5))
        plt.plot(area_data['Frame'], area_data['Avg_Intensity'], label=f'Region [{best_row}, {best_col}]', color='#007acc')
        plt.title(f'Activity Over Time (Most Active Sector: Row {best_row}, Col {best_col})')
        plt.xlabel('Frame')
        plt.ylabel('Avg Intensity')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'activity_chart.png'))
        plt.close()
        print("    Zapisano activity_chart.png")

    # --- URUCHOMIENIE CAŁOŚCI ---
    def run_all(self):
        start_time = time.time()
        try:
            self.run_video_generation()
            self.run_parameter_comparison()
            self.run_resolution_analysis()
            self.run_statistics_and_plots()
        except Exception as e:
            print(f"\nWYSTĄPIŁ BŁĄD KRYTYCZNY: {e}")
        
        total_time = time.time() - start_time
        print(f"\n--- ZAKOŃCZONO ---")
        print(f"Całkowity czas: {total_time:.2f}s")
        print(f"Wszystkie wyniki znajdują się w: {self.output_folder}")

if __name__ == "__main__":
    # Można zmienić nazwę pliku wideo tutaj
    analyzer = OpticalFlowSuite(video_filename='Highway.mp4')
    analyzer.run_all()