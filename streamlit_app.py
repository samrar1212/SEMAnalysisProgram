import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import filters, morphology, measure, segmentation
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.morphology import skeletonize, disk, binary_closing, remove_small_objects
import pandas as pd
from PIL import Image
import io

# Try to import cv2, fallback if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class StreamlitFiberAnalyzer:
    def __init__(self):
        self.original_image = None
        self.cropped_image = None
        self.fiber_mask = None
        self.all_diameters = []
        self.scale_factor = 1.0
        self.results = {}
        
    def convert_to_grayscale(self, image):
        """Convert image to grayscale with fallback methods"""
        if len(image.shape) == 3:
            if CV2_AVAILABLE:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                # Fallback: use numpy weighted average
                return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        return image
    
    def detect_scale_bar(self, image, magnification, scale_value_um):
        """Detect scale bar in SEM image and compute scale factor"""
        try:
            # Convert to grayscale if needed
            gray_img = self.convert_to_grayscale(image)
            if gray_img.max() > 1:
                gray_img = gray_img / 255.0
            
            # Adjust edge detection threshold based on magnification
            if magnification >= 5000:
                low_threshold, high_threshold = 0.85, 0.95
            elif magnification >= 2000:
                low_threshold, high_threshold = 0.75, 0.92
            elif magnification >= 1000:
                low_threshold, high_threshold = 0.5, 0.7
            else:
                low_threshold, high_threshold = 0.7, 0.9
            
            # Apply Canny edge detection
            if CV2_AVAILABLE:
                edges = cv2.Canny((gray_img * 255).astype(np.uint8), 
                                int(low_threshold * 255), 
                                int(high_threshold * 255))
                
                # Detect lines using HoughLinesP
                lines = cv2.HoughLinesP(edges, 
                                      rho=1, 
                                      theta=np.pi/180, 
                                      threshold=50,
                                      minLineLength=30,
                                      maxLineGap=10)
            else:
                # Fallback edge detection using skimage
                from skimage import feature
                edges = feature.canny(gray_img, sigma=1, low_threshold=low_threshold, high_threshold=high_threshold)
                
                # Simple line detection fallback
                from skimage.transform import hough_line, hough_line_peaks
                tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
                h, theta, d = hough_line(edges, theta=tested_angles)
                peaks = hough_line_peaks(h, theta, d, min_length=30, min_distance=20)
                
                # Convert to lines format similar to cv2.HoughLinesP
                lines = []
                for _, angle, dist in zip(*peaks[:10]):  # Take top 10 lines
                    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
                    y1 = (dist - gray_img.shape[1] * np.cos(angle)) / np.sin(angle)
                    if not (np.isnan(y0) or np.isnan(y1)):
                        lines.append([[0, int(y0), gray_img.shape[1], int(y1)]])
                lines = np.array(lines) if lines else None
            
            if lines is None or len(lines) == 0:
                st.warning("No lines detected for scale bar. Using estimated scale factor based on magnification.")
                # Estimate scale factor based on typical SEM scales
                estimated_scale = 1000 / magnification  # rough estimate
                return estimated_scale
            
            # Calculate line lengths and sort by length
            line_info = []
            for line in lines:
                if CV2_AVAILABLE:
                    x1, y1, x2, y2 = line[0]
                else:
                    x1, y1, x2, y2 = line
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                line_info.append({
                    'coords': (x1, y1, x2, y2),
                    'length': length
                })
            
            # Sort by length (longest first)
            line_info.sort(key=lambda x: x['length'], reverse=True)
            
            # Display detected lines for user selection
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(image if len(image.shape) == 3 else image, cmap='gray' if len(image.shape) == 2 else None)
            
            # Plot top 10 longest lines
            colors = plt.cm.rainbow(np.linspace(0, 1, min(10, len(line_info))))
            for i, (line, color) in enumerate(zip(line_info[:10], colors)):
                x1, y1, x2, y2 = line['coords']
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, 
                       label=f'Line {i+1} ({line["length"]:.1f} px)')
            
            ax.set_title('Detected Scale Bar Candidates')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            return fig, line_info
            
        except Exception as e:
            st.error(f"Scale detection failed: {str(e)}")
            return None, []
    
    def fiber_metric(self, image, scale, object_polarity='bright'):
        """Python implementation of MATLAB's fibermetric function"""
        gray_image = self.convert_to_grayscale(image)
        
        # Normalize image
        image_norm = gray_image.astype(np.float64)
        image_norm = (image_norm - image_norm.min()) / (image_norm.max() - image_norm.min())
        
        # Compute Hessian matrix
        sigma = scale / 3.0
        hxx, hxy, hyy = hessian_matrix(image_norm, sigma=sigma)
        
        # Compute eigenvalues
        eigenvals = hessian_matrix_eigvals([hxx, hxy, hyy])
        
        # For bright fibers, we want the larger eigenvalue (more negative)
        if object_polarity == 'bright':
            fiber_response = np.abs(eigenvals[0])
        else:
            fiber_response = np.abs(eigenvals[1])
        
        return fiber_response
    
    def process_fiber_image(self, image, fiber_scale, area_threshold, max_radius):
        """Process fiber image and extract diameters"""
        # Convert to grayscale
        gray_image = self.convert_to_grayscale(image)
        
        # Enhance fibers using fiber metric
        fiber_response = self.fiber_metric(image, fiber_scale, 'bright')
        
        # Adaptive thresholding
        threshold = filters.threshold_otsu(fiber_response)
        binary_mask = fiber_response > threshold
        
        # Morphological cleanup
        binary_mask = remove_small_objects(binary_mask, min_size=area_threshold)
        binary_mask = binary_closing(binary_mask, disk(2))
        
        # Distance transform and skeletonization
        distance_map = distance_transform_edt(binary_mask)
        skeleton = skeletonize(binary_mask)
        
        # Extract radii from skeleton
        skeleton_distances = distance_map * skeleton
        radii = skeleton_distances[skeleton_distances > 0]
        
        # Filter out large radii
        radii = radii[radii < max_radius]
        
        # Convert to diameters in nanometers
        diameters_nm = 2 * radii * self.scale_factor * 1000
        
        return diameters_nm, binary_mask
    
    def process_sections(self, image, fiber_scale, area_threshold, max_radius):
        """Process image in sections"""
        height, width = image.shape[:2]
        section_height = height // 2
        section_width = width // 2
        
        combined_mask = np.zeros((height, width), dtype=bool)
        all_diameters = []
        
        for i in range(2):
            for j in range(2):
                # Extract section
                row_start = i * section_height
                row_end = min((i + 1) * section_height, height)
                col_start = j * section_width
                col_end = min((j + 1) * section_width, width)
                
                section = image[row_start:row_end, col_start:col_end]
                
                # Process section
                section_diameters, section_mask = self.process_fiber_image(
                    section, fiber_scale, area_threshold, max_radius
                )
                
                # Add to combined results
                combined_mask[row_start:row_end, col_start:col_end] = section_mask
                all_diameters.extend(section_diameters)
        
        return all_diameters, combined_mask
    
    def calculate_statistics(self, diameters, binary_mask):
        """Calculate fiber statistics"""
        if len(diameters) == 0:
            return {
                'total_fibers': 0,
                'mean_diameter': 0,
                'median_diameter': 0,
                'std_diameter': 0,
                'fiber_density': 0,
                'avg_width_microns': 0
            }
        
        # Label connected components
        labeled_mask = measure.label(binary_mask, connectivity=2)
        num_fibers = labeled_mask.max()
        
        # Calculate density
        fiber_density = num_fibers / binary_mask.size
        
        # Calculate statistics
        return {
            'total_fibers': num_fibers,
            'mean_diameter': np.mean(diameters),
            'median_diameter': np.median(diameters),
            'std_diameter': np.std(diameters),
            'fiber_density': fiber_density,
            'avg_width_microns': np.mean(diameters) / 1000
        }

def main():
    st.set_page_config(page_title="Fiber Diameter Analyzer", layout="wide")
    
    st.title("🔬 Fiber Diameter Analyzer")
    st.markdown("Upload SEM images to analyze fiber diameters automatically")
    
    # Show environment info
    if not CV2_AVAILABLE:
        st.info("ℹ️ Running in limited mode. Some OpenCV features will use alternative implementations.")
    
    analyzer = StreamlitFiberAnalyzer()
    
    # Initialize session state for scale detection
    if 'scale_detected' not in st.session_state:
        st.session_state.scale_detected = False
    if 'detected_lines' not in st.session_state:
        st.session_state.detected_lines = []
    if 'computed_scale_factor' not in st.session_state:
        st.session_state.computed_scale_factor = None
    
    # Sidebar for parameters
    st.sidebar.header("Analysis Parameters")
    
    # Scale factor section
    st.sidebar.subheader("Scale Calibration")
    
    # Determine scale method and current scale
    scale_method = "Manual"
    
    # Show current scale factor being used
    if st.session_state.scale_detected and st.session_state.computed_scale_factor is not None:
        st.sidebar.success(f"✅ Using Detected Scale: {st.session_state.computed_scale_factor:.5f} μm/pixel")
        current_scale = st.session_state.computed_scale_factor
        scale_method = "Automatic Detection"
    else:
        manual_scale = st.sidebar.number_input("Manual Scale Factor (μm/pixel)", value=1.0, min_value=0.001, step=0.001, format="%.4f")
        st.sidebar.info(f"Using Manual Scale: {manual_scale:.4f} μm/pixel")
        current_scale = manual_scale
    
    # Button to reset to manual scale
    if st.session_state.scale_detected:
        if st.sidebar.button("🔄 Reset to Manual Scale"):
            st.session_state.scale_detected = False
            st.session_state.computed_scale_factor = None
            st.session_state.detected_lines = []  # Clear detected lines too
            st.rerun()
    
    # Set the scale factor that will be used for analysis
    analyzer.scale_factor = current_scale
    
    # Display which scale factor is being used
    st.info(f"📏 **Current Scale Factor: {analyzer.scale_factor:.5f} μm/pixel** ({analyzer.scale_factor * 1000:.2f} nm/pixel)")
    
    # Parameters for analysis
    st.sidebar.subheader("Analysis Settings")
    fiber_scale = st.sidebar.slider("Fiber Metric Scale", 3, 15, 7)
    area_threshold = st.sidebar.slider("Min Area Threshold", 10, 500, 50)
    max_radius = st.sidebar.slider("Max Radius Threshold", 5, 50, 15)
    crop_factor = st.sidebar.slider("Crop Factor", 0.1, 1.0, 0.9)
    use_sections = st.sidebar.checkbox("Process in sections", value=False)
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'])
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        analyzer.original_image = np.array(image)
        
        # Crop image
        height, width = analyzer.original_image.shape[:2]
        crop_height = int(height * crop_factor)
        analyzer.cropped_image = analyzer.original_image[:crop_height, :]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            st.image(analyzer.cropped_image, caption="Uploaded SEM Image", use_column_width=True)
        
        # Scale bar detection section
        st.subheader("🔍 Scale Bar Detection")
        
        scale_col1, scale_col2 = st.columns([1, 1])
        
        with scale_col1:
            magnification = st.number_input("Enter Magnification (e.g., 2000):", 
                                          min_value=100, max_value=100000, value=2000, step=100)
            
        with scale_col2:
            scale_value_um = st.number_input("Scale Bar Value (μm):", 
                                           min_value=0.001, max_value=1000.0, value=1.0, step=0.1)
        
        if st.button("🔍 Detect Scale Bar", type="secondary"):
            with st.spinner("Detecting scale bar..."):
                result = analyzer.detect_scale_bar(analyzer.original_image, magnification, scale_value_um)
                
                # Handle both return formats
                if isinstance(result, tuple) and len(result) == 2:
                    fig, line_info = result
                else:
                    # Single value returned (estimated scale factor)
                    estimated_scale = result
                    st.warning(f"Could not detect scale bar automatically. Using estimated scale factor: {estimated_scale:.5f} μm/pixel")
                    st.session_state.computed_scale_factor = estimated_scale
                    st.session_state.scale_detected = True
                    st.rerun()
                    return
                
                if fig is not None and len(line_info) > 0:
                    st.pyplot(fig)
                    st.session_state.detected_lines = line_info
                    # Don't set scale_detected to True yet - wait for confirmation
                else:
                    st.warning("Could not detect scale bar automatically. Please use manual scale factor.")
        
        # Show line selection if lines were detected
        if 'detected_lines' in st.session_state and len(st.session_state.detected_lines) > 0:
            line_info = st.session_state.detected_lines
            line_options = [f"Line {i+1} ({line['length']:.1f} px)" for i, line in enumerate(line_info[:10])]
            selected_line_idx = st.selectbox("Select which line is the scale bar:", 
                                            range(len(line_options)), 
                                            format_func=lambda x: line_options[x],
                                            key="scale_line_selection")
            
            col_confirm, col_clear = st.columns([1, 1])
            with col_confirm:
                if st.button("✅ Confirm Scale Bar Selection", type="primary"):
                    selected_line = line_info[selected_line_idx]
                    scale_bar_length = selected_line['length']
                    
                    # Compute scale factor (μm per pixel)
                    computed_scale_factor = scale_value_um / scale_bar_length
                    
                    # Store in session state
                    st.session_state.computed_scale_factor = computed_scale_factor
                    st.session_state.scale_detected = True
                    
                    st.success(f"""
                    ✅ Scale Calibration Complete!
                    - Scale bar length: {scale_bar_length:.2f} pixels
                    - Scale value: {scale_value_um:.2f} μm
                    - **Scale factor: {computed_scale_factor:.5f} μm/pixel**
                    - Equivalent: {computed_scale_factor * 1000:.2f} nm/pixel
                    """)
                    
                    # Clear the detected lines to hide the selection UI
                    st.session_state.detected_lines = []
                    
                    # Trigger rerun to update sidebar
                    st.rerun()
            
            with col_clear:
                if st.button("❌ Clear Detection", type="secondary"):
                    st.session_state.detected_lines = []
                    st.rerun()
        
        # Analysis section
        st.subheader("🔬 Fiber Analysis")
        
        # Analyze button
        if st.button("🔍 Analyze Fibers", type="primary"):
            with st.spinner("Analyzing fibers..."):
                try:
                    # Process image
                    if use_sections:
                        diameters, binary_mask = analyzer.process_sections(
                            analyzer.cropped_image, fiber_scale, area_threshold, max_radius
                        )
                    else:
                        diameters, binary_mask = analyzer.process_fiber_image(
                            analyzer.cropped_image, fiber_scale, area_threshold, max_radius
                        )
                    
                    analyzer.all_diameters = diameters
                    analyzer.fiber_mask = binary_mask
                    
                    with col2:
                        st.subheader("Binary Segmentation")
                        st.image(binary_mask.astype(np.uint8) * 255, caption="Detected Fibers", use_column_width=True)
                    
                    if len(diameters) > 0:
                        # Calculate statistics
                        results = analyzer.calculate_statistics(diameters, binary_mask)
                        analyzer.results = results
                        
                        # Display results
                        st.header("📊 Analysis Results")
                        
                        # Statistics in columns
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            st.metric("Total Fibers", f"{results['total_fibers']}")
                        
                        with stat_col2:
                            st.metric("Mean Diameter", f"{results['mean_diameter']:.1f} nm")
                        
                        with stat_col3:
                            st.metric("Median Diameter", f"{results['median_diameter']:.1f} nm")
                        
                        with stat_col4:
                            st.metric("Std Deviation", f"{results['std_diameter']:.1f} nm")
                        
                        # Detailed statistics table
                        st.subheader("📈 Detailed Statistics")
                        
                        stats_data = {
                            'Metric': [
                                'Total Fibers Detected', 'Mean Diameter (nm)', 'Mean Diameter (μm)', 
                                'Median Diameter (nm)', 'Standard Deviation (nm)', 'Min Diameter (nm)', 
                                'Max Diameter (nm)', 'Fiber Density (fibers/pixel²)', 
                                'Average Fiber Width (μm)', '25th Percentile (nm)',
                                '75th Percentile (nm)', '90th Percentile (nm)', '95th Percentile (nm)'
                            ],
                            'Value': [
                                f"{results['total_fibers']}", f"{results['mean_diameter']:.2f}",
                                f"{results['mean_diameter']/1000:.2f}", f"{results['median_diameter']:.2f}",
                                f"{results['std_diameter']:.2f}", f"{np.min(diameters):.2f}",
                                f"{np.max(diameters):.2f}", f"{results['fiber_density']:.6f}",
                                f"{results['avg_width_microns']:.2f}", f"{np.percentile(diameters, 25):.2f}",
                                f"{np.percentile(diameters, 75):.2f}", f"{np.percentile(diameters, 90):.2f}",
                                f"{np.percentile(diameters, 95):.2f}"
                            ]
                        }
                        
                        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
                        
                        # Analysis parameters used
                        st.subheader("⚙️ Analysis Parameters Used")
                        param_data = {
                            'Parameter': [
                                'Scale Factor (μm/pixel)', 'Scale Factor (nm/pixel)', 'Scale Detection Method',
                                'Fiber Metric Scale', 'Area Threshold', 'Max Radius Threshold', 
                                'Processed in Sections', 'Crop Factor'
                            ],
                            'Value': [
                                f"{analyzer.scale_factor:.5f}", f"{analyzer.scale_factor * 1000:.2f}",
                                scale_method,
                                f"{fiber_scale}", f"{area_threshold}", f"{max_radius}", 
                                f"{use_sections}", f"{crop_factor}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(param_data), use_container_width=True)
                        
                        # Visualizations
                        st.subheader("📊 Visualizations")
                        
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                        
                        # Original image
                        ax1.imshow(analyzer.cropped_image if len(analyzer.cropped_image.shape) == 3 else analyzer.cropped_image, 
                                  cmap='gray' if len(analyzer.cropped_image.shape) == 2 else None)
                        ax1.set_title('Original SEM Image')
                        ax1.axis('off')
                        
                        # Binary segmentation
                        ax2.imshow(binary_mask, cmap='gray')
                        ax2.set_title('Binary Segmentation')
                        ax2.axis('off')
                        
                        # Histogram
                        ax3.hist(diameters, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
                        ax3.set_xlabel('Fiber Diameter (nm)')
                        ax3.set_ylabel('Frequency')
                        ax3.set_title('Fiber Size Distribution')
                        ax3.grid(True, alpha=0.3)
                        
                        # Box plot
                        ax4.boxplot(diameters, vert=True)
                        ax4.set_ylabel('Diameter (nm)')
                        ax4.set_title('Diameter Distribution Box Plot')
                        ax4.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        
                        # Add statistics annotation
                        stats_text = f"Fiber density = {results['fiber_density']:.4f} fibers/pixel²\n"
                        stats_text += f"Average width = {results['avg_width_microns']:.2f} μm\n"
                        stats_text += f"Total fibers = {results['total_fibers']}"
                        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, fontsize=10, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        
                        st.pyplot(fig)
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        
                        # Create CSV data
                        diameter_df = pd.DataFrame({
                            'Diameter_nm': diameters,
                            'Diameter_um': diameters / 1000
                        })
                        
                        stats_df = pd.DataFrame([results])
                        
                        col_download1, col_download2 = st.columns(2)
                        
                        with col_download1:
                            csv_diameters = diameter_df.to_csv(index=False)
                            st.download_button(
                                "📄 Download Diameter Data (CSV)",
                                csv_diameters,
                                "fiber_diameters.csv",
                                "text/csv"
                            )
                        
                        with col_download2:
                            csv_stats = stats_df.to_csv(index=False)
                            st.download_button(
                                "📈 Download Statistics (CSV)",
                                csv_stats,
                                "fiber_statistics.csv",
                                "text/csv"
                            )
                        
                        # Accuracy testing section
                        st.subheader("🎯 Accuracy Testing")
                        
                        with st.expander("Test Analysis Accuracy"):
                            st.markdown("Enter ground truth measurements to validate the analysis:")
                            
                            ground_truth_text = st.text_area(
                                "Ground Truth Diameters (nm, comma-separated):",
                                value="300, 204, 317, 331, 181, 168, 410",
                                help="Enter manually measured fiber diameters separated by commas"
                            )
                            
                            if st.button("Calculate Accuracy"):
                                try:
                                    ground_truth = [float(x.strip()) for x in ground_truth_text.split(',')]
                                    
                                    # Compare with detected values
                                    n_compare = min(len(ground_truth), len(diameters))
                                    predicted = diameters[:n_compare]
                                    actual = ground_truth[:n_compare]
                                    
                                    # Calculate metrics
                                    absolute_errors = [abs(p - a) for p, a in zip(predicted, actual)]
                                    relative_errors = [abs(p - a) / a * 100 for p, a in zip(predicted, actual)]
                                    
                                    mae = np.mean(absolute_errors)
                                    mre = np.mean(relative_errors)
                                    rmse = np.sqrt(np.mean([(p - a)**2 for p, a in zip(predicted, actual)]))
                                    correlation = np.corrcoef(predicted, actual)[0,1] if len(predicted) > 1 else 0
                                    
                                    # Display accuracy results
                                    acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)
                                    
                                    with acc_col1:
                                        st.metric("Mean Absolute Error", f"{mae:.2f} nm")
                                    with acc_col2:
                                        st.metric("Mean Relative Error", f"{mre:.2f}%")
                                    with acc_col3:
                                        st.metric("RMSE", f"{rmse:.2f} nm")
                                    with acc_col4:
                                        st.metric("Correlation", f"{correlation:.3f}")
                                    
                                    # Individual comparisons
                                    comparison_data = {
                                        'Sample': [f"Sample {i+1}" for i in range(n_compare)],
                                        'Predicted (nm)': [f"{p:.1f}" for p in predicted],
                                        'Actual (nm)': [f"{a:.1f}" for a in actual],
                                        'Absolute Error (nm)': [f"{ae:.1f}" for ae in absolute_errors],
                                        'Relative Error (%)': [f"{re:.1f}" for re in relative_errors]
                                    }
                                    
                                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                                    
                                    # Validation plot
                                    fig_val, (ax_val1, ax_val2) = plt.subplots(1, 2, figsize=(12, 5))
                                    
                                    # Scatter plot
                                    ax_val1.scatter(actual, predicted, alpha=0.7, s=100)
                                    ax_val1.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', linewidth=2, label='Perfect Prediction')
                                    ax_val1.set_xlabel('Actual Diameter (nm)')
                                    ax_val1.set_ylabel('Predicted Diameter (nm)')
                                    ax_val1.set_title('Predicted vs Actual Diameters')
                                    ax_val1.grid(True, alpha=0.3)
                                    ax_val1.legend()
                                    
                                    # Add R² value
                                    r_squared = correlation**2
                                    ax_val1.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax_val1.transAxes, 
                                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                                    
                                    # Error distribution
                                    errors = [p - a for p, a in zip(predicted, actual)]
                                    ax_val2.hist(errors, bins=min(10, len(errors)), alpha=0.7, edgecolor='black')
                                    ax_val2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
                                    ax_val2.set_xlabel('Error (Predicted - Actual) nm')
                                    ax_val2.set_ylabel('Frequency')
                                    ax_val2.set_title('Error Distribution')
                                    ax_val2.grid(True, alpha=0.3)
                                    ax_val2.legend()
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig_val)
                                    
                                except Exception as e:
                                    st.error(f"Accuracy calculation failed: {str(e)}")
                        
                    else:
                        st.error("No fibers detected. Try adjusting the parameters:")
                        st.markdown("""
                        - Decrease the **Area Threshold** for smaller fibers
                        - Adjust the **Fiber Metric Scale** (try 5-9)
                        - Check if the scale factor is correct
                        - Try processing in sections for complex images
                        """)
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.exception(e)
    
    # Instructions
    with st.expander("ℹ️ How to Use This Analyzer"):
        st.markdown("""
        ### Step-by-Step Instructions:
        
        1. **Upload Image**: Select your SEM image using the file uploader
        
        2. **Scale Calibration** (Choose one method):
           - **Automatic**: Enter magnification and scale bar value, then click "Detect Scale Bar"
           - **Manual**: Enter the known scale factor directly
        
        3. **Adjust Parameters**:
           - **Fiber Metric Scale**: Controls fiber detection sensitivity (5-9 works well for most images)
           - **Area Threshold**: Minimum fiber area to detect (increase for noisy images)
           - **Max Radius**: Maximum fiber radius to consider
           - **Crop Factor**: Portion of image to analyze (0.9 = top 90%)
           - **Process in Sections**: Enable for large or complex images
        
        4. **Run Analysis**: Click "Analyze Fibers" to process the image
        
        5. **Review Results**: Check statistics, visualizations, and download data
        
        6. **Validate (Optional)**: Use the accuracy testing section with known measurements
        
        ### Tips for Best Results:
        - Use high-quality SEM images with good contrast
        - Ensure scale bars are visible and well-defined
        - For dense fiber networks, try increasing the area threshold
        - For thin fibers, decrease the fiber metric scale
        - Process in sections for large or complex images
        - Typical parameter ranges:
          - Fiber Metric Scale: 5-9
          - Area Threshold: 25-100
          - Max Radius: 10-20

        ### Parameter Sensitivity Guidelines:
        - **High magnification images** (>5000x): Use smaller fiber metric scale (3-5)
        - **Low magnification images** (<1000x): Use larger fiber metric scale (7-12)
        - **Dense fiber mats**: Increase area threshold to reduce noise
        - **Sparse fibers**: Decrease area threshold to catch small features
        - **Thick fibers**: Increase max radius threshold
        - **Thin nanofibers**: Keep max radius low (5-15)
        
        ### Troubleshooting:
        - **No fibers detected**: Lower area threshold, adjust fiber metric scale
        - **Too many false positives**: Increase area threshold, use sections processing
        - **Scale bar not detected**: Use manual scale factor or adjust magnification/scale values
        - **Inaccurate measurements**: Verify scale factor, try different fiber metric scales
        - **Analysis too slow**: Enable section processing, reduce image size with crop factor
        
        ### Expected Input Formats:
        - **Image types**: PNG, JPG, JPEG, TIFF, BMP
        - **Image quality**: High resolution SEM images with clear fiber structures
        - **Scale information**: Either visible scale bar or known magnification
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About This Tool
    This fiber diameter analyzer uses advanced image processing techniques including:
    - Hessian-based fiber enhancement (fibermetric)
    - Adaptive thresholding and morphological operations
    - Distance transform and skeletonization for diameter measurement
    - Automatic scale bar detection using edge detection and line detection
    
    **Note**: Results should always be validated against manual measurements for your specific application.
    """)

if __name__ == "__main__":
    main()
