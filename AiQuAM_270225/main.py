import os
import argparse
from Q3_open_source_fit_two_half_gaussians import printIQreport

def main():
    print("Welcome to the IQ Analysis Application!")
    print("This program allows you to analyze .raw files and generate reports.")
    
    parser = argparse.ArgumentParser(description="IQ Analysis Application")
    parser.add_argument("file_path", type=str, help="Path to the .raw file to analyze.")
    parser.add_argument("save_path", type=str, help="Directory where the report will be saved.")
    parser.add_argument("--report_type", choices=['console', 'pdf'], default='pdf', 
                        help="Specify the type of report to generate: 'console' for in-console report or 'pdf' for PDF report.")
    parser.add_argument("--number_of_slices", type=int, default=5, 
                        help="Number of slices to analyze (default: 5).")
    parser.add_argument("--margin", type=int, default=10, 
                        help="Margin for analysis (default: 10).")
    parser.add_argument("--min_particle_size", type=int, default=None, 
                        help="Minimum particle size for analysis (default: None).")
    parser.add_argument("--max_particle_size", type=int, default=None, 
                        help="Maximum particle size for analysis (default: None).")
    parser.add_argument("--pores_to_analyze", type=int, default=3, 
                        help="Number of pores to analyze (default: 3).")
    parser.add_argument("--cnr_mask_erosion_value", type=float, default=None, 
                        help="Erosion value for CNR mask (optional).")
    parser.add_argument("--bright_pores", action='store_true', 
                        help="Analyze bright pores.")
    parser.add_argument("--local_tomo", action='store_true', 
                        help="Use local tomography.")

    args = parser.parse_args()

    if not os.path.isfile(args.file_path):
        print(f"Error: The file '{args.file_path}' does not exist.")
        return

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.report_type == 'pdf':
        printIQreport(
            file_path=args.file_path,
            mask_path=None,
            save_path=args.save_path,
            number_of_slices=args.number_of_slices,
            margin=args.margin,
            min_particle_size=args.min_particle_size,
            max_particle_size=args.max_particle_size,
            Pores_to_analyze=args.pores_to_analyze,
            cnr_mask_erosion_value=args.cnr_mask_erosion_value,
            bright_pores=args.bright_pores,
            local_tomo=args.local_tomo
        )
        print("PDF report generated successfully.")
    elif args.report_type == 'console':
        # Implement console report generation logic here
        print("Console report generation is not yet implemented.")

if __name__ == "__main__":
    main()