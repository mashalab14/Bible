# Bible App Tone & Style Guide

## 1. Overall Voice
- **Tone**: Warm, calm, steady, trustworthy.  
- **Personality**: A thoughtful companion, not a coach or drill sergeant.  
- **Approach**: Encourage, invite, affirm. Avoid guilt, hype, or pressure.  
- **Consistency**: Every message (verse reflection, milestone, error message) should feel like it comes from the same calm presence.  

---

## 2. Language Rules
- **Sentence length**: Max 18 words. Use line breaks for readability on small screens.  
- **Vocabulary**: Everyday English (calm, peace, kindness, gratitude, steady, gentle, thankful).  
- **Avoid**: slang, theology jargon, sarcasm, political hints.  
- **Voice**: 2nd person (“you”), inclusive (“we”). Example: *“We’re grateful you returned today.”*  
- **Affirmations**: Always thank the user for showing up. Example: *“Thank you for spending a moment here.”*  
- **Actionable nudges**: End reflections with a gentle suggestion, not a command. Example: *“Take a slow breath before you move on.”*  

---

## 3. Content Structure
### Verse Display
- Serif font, large (24–28px).  
- High contrast text on calm background.  
- Always show reference (e.g., *1 Peter 5:7*).  

### Reflection
- 3–4 sentences.  
- Start by acknowledging emotion (anxious, joyful, reflective).  
- Give one plain, supportive thought.  
- End with one practical nudge.  

### Journal Prompt
- One open-ended question, one sentence only.  
- Never leading, never judgmental.  
- Example: *“What small worry could you let go of today?”*  

### Milestone Messages
- Gratitude first (*“Thank you for being here”*).  
- Recognition second (*“Seven days of presence”*).  
- Tone: gentle celebration, not achievement.  
- Animation: slow fade in/out, no confetti, no loud colors.  

---

## 4. UX Writing Patterns
- **Buttons**  
  - Short, action-oriented, plain English.  
  - Examples: *“Listen”*, *“Read Full Chapter”*, *“Save to Journal”*, *“Test Your Knowledge”*.  
  - Avoid: *“Go”*, *“Smash it”*, *“Level up”*, *“Unlock”*.  

- **Notifications**  
  - Soft, invitational, no pressure.  
  - Examples:  
    - *“A gentle verse is waiting for you today.”*  
    - *“Take a quiet moment with today’s reflection.”*  
  - Avoid: *“Don’t lose your streak!”*  

- **Error States**  
  - Calm, reassuring.  
  - Example: *“Something didn’t load. Try again when you’re ready.”*  

---

## 5. Visual Style
- **Typography**  
  - Verses: Serif (Newsreader, Georgia, etc.).  
  - UI: Sans-serif (Noto Sans, Inter).  
  - No novelty fonts.  

- **Colors**  
  - Muted, calm palette (navy, deep green, warm beige, soft gray).  
  - Backgrounds: gradients or photos (sunrise, sea, sky, forest).  
  - Never neon, never harsh contrasts beyond accessibility needs.  

- **Layout**  
  - Generous spacing.  
  - Max width for text (no walls of words).  
  - Clear visual hierarchy: Verse → Reflection → Journal.  

---

## 6. Audio & Guided Content
- **Voice**: Neutral American or British, steady, gentle pacing.  
- **Tone**: Slightly slower than normal speech, pauses between sentences.  
- **Background**: Optional soft ambient (piano, strings, nature). Must not overpower voice.  
- **Length**: 60–90 seconds max for guided reflections.  

---

## 7. Cultural Sensitivity
- **Bible Versions**: Stick to widely accepted translations (KJV, ESV, NIV). Avoid niche/controversial.  
- **Values**: emphasize peace, hope, gratitude, resilience.  
- **Avoid**: political references, culture-war verses, or themes that alienate (violence, slavery, sexism).  

---

## 8. Recognition & Gratitude
- **Streak replacement**: Call it *Milestones* or *Journey markers*.  
- **Language**: *“Thank you for being here”*, *“Your presence is a gift”*, *“We’re grateful you returned.”*  
- **Emotion**: smile, warmth, not hype.  
- **Frequency**: once when milestone reached, never nag.  

---

## 9. Accessibility
- **Font size toggle** (default large).  
- **High contrast option**.  
- **Dark mode**: default to user device preference.  
- **Readable at arm’s length**: test on small phone.  

---

## 10. Internal Rules (for dev/design team)
- Never ship copy that isn’t tested aloud (must *sound calm*).  
- Always run text through readability check (target Grade 7 or lower).  
- Every feature must pass: *Is this calming? Is this respectful?*  
- If in doubt, choose **simpler, calmer, smaller**.  

---

👉 Save this file as **`STYLE_GUIDE.md`** in the repo. Everyone — devs, designers, writers — should follow it to keep the app’s experience consistent.
