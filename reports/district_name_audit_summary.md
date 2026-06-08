# District Name Audit Summary

This audit flags historical district-name issues before district crosswalk construction.
No geocoding, map-layer download, coordinate assignment, or manual district-name correction was performed.

## Counts

- Input district rows: 727
- States: 35
- Unique district_id values: 727
- Problematic district rows: 128
- High priority rows: 1
- Medium priority rows: 127
- Low priority rows: 599
- Normalized conflict groups: 0

## High-Priority Names

- `Punjab` / `S`: flag_name_too_short, flag_single_character_name, flag_possible_truncation

## District Names Appearing In Multiple States

- `ADILABAD`: names `ADILABAD`; states `Andhra Pradesh; Telangana`
- `ALMORA`: names `ALMORA`; states `Uttar Pradesh; Uttarakhand`
- `AURANGABAD`: names `AURANGABAD`; states `Bihar; Maharashtra`
- `BAGESHWAR`: names `BAGESHWAR`; states `Uttar Pradesh; Uttarakhand`
- `BALRAMPUR`: names `BALRAMPUR`; states `Chhattisgarh; Uttar Pradesh`
- `BASTAR`: names `BASTAR`; states `Chhattisgarh; Madhya Pradesh`
- `BIJAPUR`: names `BIJAPUR`; states `Chhattisgarh; Karnataka`
- `BILASPUR`: names `BILASPUR`; states `Chhattisgarh; Himachal Pradesh; Madhya Pradesh`
- `BOKARO`: names `BOKARO`; states `Bihar; Jharkhand`
- `CHAMOLI`: names `CHAMOLI`; states `Uttar Pradesh; Uttarakhand`
- `CHAMPAWAT`: names `CHAMPAWAT`; states `Uttar Pradesh; Uttarakhand`
- `CHATRA`: names `CHATRA`; states `Bihar; Jharkhand`
- `DANTEWADA`: names `DANTEWADA`; states `Chhattisgarh; Madhya Pradesh`
- `DEHRADUN`: names `DEHRADUN`; states `Uttar Pradesh; Uttarakhand`
- `DEOGHAR`: names `DEOGHAR`; states `Bihar; Jharkhand`
- `DHAMTARI`: names `DHAMTARI`; states `Chhattisgarh; Madhya Pradesh`
- `DHANBAD`: names `DHANBAD`; states `Bihar; Jharkhand`
- `DUMKA`: names `DUMKA`; states `Bihar; Jharkhand`
- `DURG`: names `DURG`; states `Chhattisgarh; Madhya Pradesh`
- `EAST SINGHBUM`: names `EAST SINGHBUM`; states `Bihar; Jharkhand`
- `GARHWA`: names `GARHWA`; states `Bihar; Jharkhand`
- `GIRIDIH`: names `GIRIDIH`; states `Bihar; Jharkhand`
- `GODDA`: names `GODDA`; states `Bihar; Jharkhand`
- `GUMLA`: names `GUMLA`; states `Bihar; Jharkhand`
- `HAMIRPUR`: names `HAMIRPUR`; states `Himachal Pradesh; Uttar Pradesh`
- `HARIDWAR`: names `HARIDWAR`; states `Uttar Pradesh; Uttarakhand`
- `HAZARIBAGH`: names `HAZARIBAGH`; states `Bihar; Jharkhand`
- `HYDERABAD`: names `HYDERABAD`; states `Andhra Pradesh; Telangana`
- `JANJGIRCHAMPA`: names `JANJGIR-CHAMPA`; states `Chhattisgarh; Madhya Pradesh`
- `JASHPUR`: names `JASHPUR`; states `Chhattisgarh; Madhya Pradesh`
- `KABIRDHAM`: names `KABIRDHAM`; states `Chhattisgarh; Madhya Pradesh`
- `KANKER`: names `KANKER`; states `Chhattisgarh; Madhya Pradesh`
- `KARIMNAGAR`: names `KARIMNAGAR`; states `Andhra Pradesh; Telangana`
- `KHAMMAM`: names `KHAMMAM`; states `Andhra Pradesh; Telangana`
- `KODERMA`: names `KODERMA`; states `Bihar; Jharkhand`
- `KORBA`: names `KORBA`; states `Chhattisgarh; Madhya Pradesh`
- `KOREA`: names `KOREA`; states `Chhattisgarh; Madhya Pradesh`
- `LOHARDAGA`: names `LOHARDAGA`; states `Bihar; Jharkhand`
- `MAHASAMUND`: names `MAHASAMUND`; states `Chhattisgarh; Madhya Pradesh`
- `MAHBUBNAGAR`: names `MAHBUBNAGAR`; states `Andhra Pradesh; Telangana`
- `MEDAK`: names `MEDAK`; states `Andhra Pradesh; Telangana`
- `NAINITAL`: names `NAINITAL`; states `Uttar Pradesh; Uttarakhand`
- `NALGONDA`: names `NALGONDA`; states `Andhra Pradesh; Telangana`
- `NIZAMABAD`: names `NIZAMABAD`; states `Andhra Pradesh; Telangana`
- `PAKUR`: names `PAKUR`; states `Bihar; Jharkhand`
- `PALAMU`: names `PALAMU`; states `Bihar; Jharkhand`
- `PAURI GARHWAL`: names `PAURI GARHWAL`; states `Uttar Pradesh; Uttarakhand`
- `PITHORAGARH`: names `PITHORAGARH`; states `Uttar Pradesh; Uttarakhand`
- `PRATAPGARH`: names `PRATAPGARH`; states `Rajasthan; Uttar Pradesh`
- `RAIPUR`: names `RAIPUR`; states `Chhattisgarh; Madhya Pradesh`
- `RAJNANDGAON`: names `RAJNANDGAON`; states `Chhattisgarh; Madhya Pradesh`
- `RANCHI`: names `RANCHI`; states `Bihar; Jharkhand`
- `RANGAREDDI`: names `RANGAREDDI`; states `Andhra Pradesh; Telangana`
- `RUDRA PRAYAG`: names `RUDRA PRAYAG`; states `Uttar Pradesh; Uttarakhand`
- `SAHEBGANJ`: names `SAHEBGANJ`; states `Bihar; Jharkhand`
- `SURGUJA`: names `SURGUJA`; states `Chhattisgarh; Madhya Pradesh`
- `TEHRI GARHWAL`: names `TEHRI GARHWAL`; states `Uttar Pradesh; Uttarakhand`
- `UDAM SINGH NAGAR`: names `UDAM SINGH NAGAR`; states `Uttar Pradesh; Uttarakhand`
- `UTTAR KASHI`: names `UTTAR KASHI`; states `Uttar Pradesh; Uttarakhand`
- `WARANGAL`: names `WARANGAL`; states `Andhra Pradesh; Telangana`
- `WEST SINGHBHUM`: names `WEST SINGHBHUM`; states `Bihar; Jharkhand`

## Raw Names Sharing The Same Normalized Key

- None.

## Input Integrity

- Input SHA-256 before audit: `550D2EF417D736B9DDFA7870A2064A74284EFC440A68F5D01660407237DA8911`
- Input SHA-256 after audit: `550D2EF417D736B9DDFA7870A2064A74284EFC440A68F5D01660407237DA8911`
- Input file was not modified by this audit.
