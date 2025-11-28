import yaml
import os
from random import shuffle

sample_size = 100

def create_file_mapping():
    """Create mapping between descriptions_commands files and cultures metadata files."""
    mapping = {
        'AskAChinese_descriptions_commands.yaml': 'cul_China/metadata.yaml',
        'AskAGerman_descriptions_commands.yaml': 'cul_Germany/metadata.yaml',
        'AskAJapanese_descriptions_commands.yaml': 'cul_Japan/metadata.yaml',
        'AskAnAfrican_descriptions_commands.yaml': 'cul_Africa/metadata.yaml',
        'AskAnAmerican_descriptions_commands.yaml': 'cul_US/metadata.yaml',
        'AskArgentina_descriptions_commands.yaml': 'cul_Argentina/metadata.yaml',
        'AskARussian_descriptions_commands.yaml': 'cul_Russia/metadata.yaml',
        'askasia_descriptions_commands.yaml': 'cul_Asia/metadata.yaml',
        'AskBalkans_descriptions_commands.yaml': 'cul_Balkans/metadata.yaml',
        'askegypt_descriptions_commands.yaml': 'cul_Egypt/metadata.yaml',
        'AskEurope_descriptions_commands.yaml': 'cul_Europe/metadata.yaml',
        'AskFrance_descriptions_commands.yaml': 'cul_France/metadata.yaml',
        'AskIndia_descriptions_commands.yaml': 'cul_India/metadata.yaml',
        'askitaly_descriptions_commands.yaml': 'cul_Italy/metadata.yaml',
        'asklatinamerica_descriptions_commands.yaml': 'cul_Latinamerica/metadata.yaml',
        'askmexico_descriptions_commands.yaml': 'cul_Mexico/metadata.yaml',
        'AskMiddleEast_descriptions_commands.yaml': 'cul_MiddleEast/metadata.yaml',
        'AskPH_descriptions_commands.yaml': 'cul_PH/metadata.yaml',
        'AskSouthAfrica_descriptions_commands.yaml': 'cul_Southafrica/metadata.yaml',
        'AskTurkey_descriptions_commands.yaml': 'cul_Turkey/metadata.yaml',
        'AskUK_descriptions_commands.yaml': 'cul_UK/metadata.yaml',
    }
    return mapping

def read_descriptions_commands(file_path):
    """Read descriptions_commands file and extract text entries."""
    descriptions = []
    current_description = ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()  # Remove trailing whitespace but keep leading spaces
                
                if line.startswith('- '):
                    # If we have a previous description, save it
                    if current_description:
                        descriptions.append(current_description.strip())
                    # Start new description
                    current_description = line[2:]  # Remove '- '
                elif line.startswith('  ') and current_description:
                    # This is a continuation line, add it to current description
                    current_description += " " + line.strip()
                elif line.strip() == "":
                    # Empty line, ignore
                    continue
                else:
                    # Unexpected line format, ignore
                    continue
            
            # Don't forget the last description
            if current_description:
                descriptions.append(current_description.strip())
                
        return descriptions
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def read_metadata_yaml(file_path):
    """Read existing metadata.yaml file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def write_metadata_yaml(file_path, data):
    """Write metadata.yaml file with updated descriptions."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False

def process_file_pair(descriptions_file, metadata_file, base_path):
    """Process a single pair of descriptions_commands and metadata files."""
    descriptions_path = os.path.join(base_path, 'data', 'descriptions_commands', descriptions_file)
    metadata_path = os.path.join(base_path, 'cultures', metadata_file)
    
    print(f"Processing: {descriptions_file} -> {metadata_file}")
    
    # Check if files exist
    if not os.path.exists(descriptions_path):
        print(f"  Warning: {descriptions_path} not found")
        return False
    
    if not os.path.exists(metadata_path):
        print(f"  Warning: {metadata_path} not found")
        return False
    
    # Read descriptions from commands file
    descriptions = read_descriptions_commands(descriptions_path)
    if not descriptions:
        print(f"  Warning: No descriptions found in {descriptions_file}")
        return False
    
    # Read existing metadata
    metadata = read_metadata_yaml(metadata_path)
    if metadata is None:
        print(f"  Error: Could not read {metadata_file}")
        return False
    
    # Update descriptions in metadata
    metadata['descriptions'] = descriptions
    
    # Write updated metadata
    if write_metadata_yaml(metadata_path, metadata):
        print(f"  Success: Updated {len(descriptions)} descriptions in {metadata_file}")
        return True
    else:
        print(f"  Error: Failed to write {metadata_file}")
        return False

def move_descriptions_to_metadata():
    """Move descriptions from descriptions_commands files to cultures metadata files."""
    # Get the script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    print(f"Project root: {project_root}")
    
    # Create file mapping
    mapping = create_file_mapping()
    
    # Process each file pair
    success_count = 0
    total_count = len(mapping)
    
    for descriptions_file, metadata_file in mapping.items():
        if process_file_pair(descriptions_file, metadata_file, project_root):
            success_count += 1
        print()  # Add blank line for readability
    
    print(f"Summary: {success_count}/{total_count} files processed successfully")

DATA_DIR="cul_data"

descriptions_paths = [f for f in os.listdir(os.path.join(DATA_DIR, "descriptions"))]
for descriptions_path in descriptions_paths:
    print(descriptions_path)
    descriptions_path = os.path.join(DATA_DIR, "descriptions", descriptions_path)
    commands_path = descriptions_path.replace("descriptions", "commands")
    if not os.path.exists(commands_path):
        print(f"skipped {commands_path}")
        break
    with open(descriptions_path, "r") as f:
        descriptions_list = yaml.safe_load(f)
    with open(commands_path, "r") as f:
        commands_list = yaml.safe_load(f)
    subreddit = os.path.basename(descriptions_path).replace("_descriptions.yaml", "")
    print(f"processing {subreddit}")
    merged_commands_descriptions = [command + " " + description for command, description in zip(commands_list[:sample_size], descriptions_list[:sample_size])]
    commands = commands_list[sample_size:]
    descs = descriptions_list[sample_size:]

    shuffle(commands)
    shuffle(descs)

    full_list = merged_commands_descriptions + descs + commands
    shuffle(full_list)
    with open(os.path.join(DATA_DIR, "descriptions_commands", f"{subreddit}_descriptions_commands.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(full_list, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

# After merging all descriptions_commands files, move them to cultures metadata
print("\n" + "="*50)
print("Moving descriptions to cultures metadata files...")
print("="*50)
move_descriptions_to_metadata()