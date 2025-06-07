# 🚀 Community Collection System - Quick Start Guide

## **TL;DR - How to Use**

### 1. **Collect Data** 📊
- **Option A**: Use the HTML form at `data/community/contribution_form.html`
- **Option B**: Create Google Form with the required fields (see below)
- **Goal**: Collect 200 V60 brewing experiences from the community

### 2. **Outreach** 📢
Use the pre-written templates in `data/community/`:
- `reddit_post_template.txt` - For r/coffee, r/espresso
- `discord_message_template.txt` - For Discord coffee servers
- `slack_message_template.txt` - For Slack coffee communities
- `coffee_shop_email_template.txt` - For professional baristas

### 3. **Process Data** ⚙️
```bash
# 1. Export CSV from Google Forms to data/community/responses.csv
# 2. Run the processing script
python scripts/process_community_data.py
```

### 4. **Integration** 🔗
The processed JSON can be merged with existing training data for model improvement.

---

## **Required Form Fields**

| Field | Type | Options/Format |
|-------|------|----------------|
| `coffee_amount` | Number | 10-30g |
| `water_amount` | Number | 150-500g |
| `grind_size` | Select | very_fine, fine, medium_fine, medium, medium_coarse, coarse |
| `brew_time` | Text | mm:ss format (e.g., "3:30") |
| `taste_notes` | Textarea | Free text description |
| `adjustment` | Select | much_finer, finer, slightly_finer, no_change, slightly_coarser, coarser, much_coarser |
| `reasoning` | Textarea | Why this adjustment would help |

---

## **Example Workflow**

1. **Deploy Form**: Share `data/community/contribution_form.html` or create Google Form
2. **Outreach**: Post to Reddit using `reddit_post_template.txt`
3. **Collect**: Wait for responses (aim for 50-75 from each platform)
4. **Process**: Run `python scripts/process_community_data.py`
5. **Review**: Check output JSON file for quality
6. **Repeat**: Continue until you have 200+ examples

---

## **Quality Assurance**

The system automatically:
- ✅ Validates all required fields are present
- ✅ Maps grind adjustments to standard format
- ✅ Determines extraction category from taste notes
- ✅ Calculates expected brew times
- ✅ Assigns confidence scores
- ✅ Filters out invalid entries

---

## **Target Distribution**

- **Reddit communities**: 50-75 examples
- **Discord/Slack groups**: 25-50 examples
- **Coffee shop baristas**: 75-100 examples
- **Other forums**: 25-50 examples
- **Total Goal**: 200 high-quality examples

---

## **Files Overview**

```
data/community/
├── contribution_form.html          # Ready-to-deploy web form
├── example_responses.csv           # Sample data format
├── reddit_post_template.txt        # Reddit outreach template
├── discord_message_template.txt    # Discord outreach template
├── slack_message_template.txt      # Slack outreach template
├── coffee_shop_email_template.txt  # Email template for shops
└── README.md                       # Detailed documentation

scripts/
├── setup_community_collection.py   # Initial setup script
├── test_community_processing.py    # Test processing functionality
└── process_community_data.py       # Main processing workflow

src/
└── community_collection.py         # Core processing module
```

---

## **Need Help?**

- Check `data/community/README.md` for detailed documentation
- Run `python scripts/test_community_processing.py` to test functionality
- Use `data/community/example_responses.csv` as a reference format
